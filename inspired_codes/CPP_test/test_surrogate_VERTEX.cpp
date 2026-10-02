// ============================================================
// GINN TORCHSCRIPT INFERENCE TEST — GEOMETRY-AWARE MODEL
// ------------------------------------------------------------
//
// INPUT:
//   training_data.csv
//   overlap_model.pt
//   scalers.dat
//
// OUTPUT:
//   predictions.csv
//
// Required input columns:
//
//   position_z
//   Q_orientation.w()
//   Q_orientation.x()
//   Q_orientation.y()
//   Q_orientation.z()
//
// Optional input columns:
//
//   overlap(m)
//   normal_x
//   normal_y
//   normal_z
//
// All additional input columns are preserved automatically.
// ============================================================

#include <torch/script.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

// ============================================================
// GLOBAL SCALER DATA
// ============================================================
//
// [0] = log_overlap
// [1] = nx
// [2] = ny
// [3] = nz
// ============================================================

std::vector<double> y_mean;
std::vector<double> y_scale;

double eps = 1e-15;

// ============================================================
// GEOMETRY CONSTANTS
// ============================================================

static constexpr double CUBE_SIZE = 1.0;
static constexpr double HALF = CUBE_SIZE / 2.0;
static constexpr double PI = 3.1415926535897932384626433832795;

// ============================================================
// CSV SPLIT
// ============================================================
//
// This parser assumes that fields do not contain quoted commas.
// ============================================================

std::vector<std::string> split(
    const std::string& line
    )
{
    std::stringstream ss(line);
    std::string item;
    std::vector<std::string> tokens;

    while(std::getline(ss, item, ','))
    {
        tokens.push_back(item);
    }

    // Preserve a final empty field.
    if(!line.empty() && line.back() == ',')
    {
        tokens.push_back("");
    }

    return tokens;
}

// ============================================================
// CSV JOIN
// ============================================================

std::string join_csv(
    const std::vector<std::string>& tokens
    )
{
    std::ostringstream output;

    for(size_t i = 0; i < tokens.size(); ++i)
    {
        if(i > 0)
        {
            output << ",";
        }

        output << tokens[i];
    }

    return output.str();
}

// ============================================================
// TRIM STRING
// ============================================================

void trim_string(
    std::string& value
    )
{
    // Remove Windows carriage return.
    if(!value.empty() && value.back() == '\r')
    {
        value.pop_back();
    }

    // Remove leading whitespace.
    while(
        !value.empty() &&
        std::isspace(
            static_cast<unsigned char>(value.front())
            )
        )
    {
        value.erase(value.begin());
    }

    // Remove trailing whitespace.
    while(
        !value.empty() &&
        std::isspace(
            static_cast<unsigned char>(value.back())
            )
        )
    {
        value.pop_back();
    }
}

// ============================================================
// SAFE STRING TO DOUBLE
// ============================================================

bool try_parse_double(
    const std::string& text,
    double& value
    )
{
    std::string cleaned = text;
    trim_string(cleaned);

    if(cleaned.empty())
    {
        return false;
    }

    try
    {
        size_t processed_characters = 0;

        value = std::stod(
            cleaned,
            &processed_characters
            );

        if(processed_characters != cleaned.size())
        {
            return false;
        }

        return std::isfinite(value);
    }
    catch(const std::exception&)
    {
        return false;
    }
}

// ============================================================
// LOAD SCALER
// ============================================================

bool load_scaler(
    const std::string& filename
    )
{
    std::ifstream file(filename);

    if(!file.is_open())
    {
        std::cerr
            << "Could not open scaler file: "
            << filename
            << "\n";

        return false;
    }

    y_mean.clear();
    y_scale.clear();

    std::string line;

    while(std::getline(file, line))
    {
        trim_string(line);

        if(line == "y_mean")
        {
            if(!std::getline(file, line))
            {
                std::cerr << "Missing y_mean values\n";
                return false;
            }

            trim_string(line);

            std::stringstream ss(line);
            double value;

            while(ss >> value)
            {
                y_mean.push_back(value);
            }
        }
        else if(line == "y_scale")
        {
            if(!std::getline(file, line))
            {
                std::cerr << "Missing y_scale values\n";
                return false;
            }

            trim_string(line);

            std::stringstream ss(line);
            double value;

            while(ss >> value)
            {
                y_scale.push_back(value);
            }
        }
        else if(line == "eps")
        {
            if(!std::getline(file, line))
            {
                std::cerr << "Missing eps value\n";
                return false;
            }

            trim_string(line);

            try
            {
                eps = std::stod(line);
            }
            catch(const std::exception&)
            {
                std::cerr << "Invalid eps value\n";
                return false;
            }
        }
    }

    file.close();

    if(y_mean.size() != 4 || y_scale.size() != 4)
    {
        std::cerr
            << "Invalid scaler data.\n"
            << "Expected four y_mean and four y_scale values:\n"
            << "log_overlap nx ny nz\n";

        return false;
    }

    for(double scale : y_scale)
    {
        if(scale == 0.0 || !std::isfinite(scale))
        {
            std::cerr
                << "Invalid scaler data: y_scale contains zero "
                << "or non-finite values.\n";

            return false;
        }
    }

    return true;
}

// ============================================================
// QUATERNION ROTATION OF A 3D POINT
// ============================================================

inline void rotate_point_by_quat(
    double qw,
    double qx,
    double qy,
    double qz,
    double px,
    double py,
    double pz,
    double& out_x,
    double& out_y,
    double& out_z
    )
{
    const double R00 =
        1.0 - 2.0 * (qy * qy + qz * qz);

    const double R01 =
        2.0 * (qx * qy - qz * qw);

    const double R02 =
        2.0 * (qx * qz + qy * qw);

    const double R10 =
        2.0 * (qx * qy + qz * qw);

    const double R11 =
        1.0 - 2.0 * (qx * qx + qz * qz);

    const double R12 =
        2.0 * (qy * qz - qx * qw);

    const double R20 =
        2.0 * (qx * qz - qy * qw);

    const double R21 =
        2.0 * (qy * qz + qx * qw);

    const double R22 =
        1.0 - 2.0 * (qx * qx + qy * qy);

    out_x = R00 * px + R01 * py + R02 * pz;
    out_y = R10 * px + R11 * py + R12 * pz;
    out_z = R20 * px + R21 * py + R22 * pz;
}

// ============================================================
// EXACT ANALYTICAL CUBE-PLANE OVERLAP AND NORMAL
// ============================================================

inline void compute_exact_overlap_and_normal(
    double qw,
    double qx,
    double qy,
    double qz,
    double position_z,
    double& out_overlap,
    double& out_nx,
    double& out_ny,
    double& out_nz
    )
{
    static const std::array<
        std::array<double, 3>,
        8
        > local_vertices =
        {{
            {{-HALF, -HALF, -HALF}},
            {{-HALF, -HALF,  HALF}},
            {{-HALF,  HALF, -HALF}},
            {{-HALF,  HALF,  HALF}},
            {{ HALF, -HALF, -HALF}},
            {{ HALF, -HALF,  HALF}},
            {{ HALF,  HALF, -HALF}},
            {{ HALF,  HALF,  HALF}}
        }};

    std::array<
        std::array<double, 3>,
        8
        > world_vertices;

    double min_world_z =
        std::numeric_limits<double>::infinity();

    for(size_t i = 0; i < local_vertices.size(); ++i)
    {
        double rx;
        double ry;
        double rz;

        rotate_point_by_quat(
            qw,
            qx,
            qy,
            qz,
            local_vertices[i][0],
            local_vertices[i][1],
            local_vertices[i][2],
            rx,
            ry,
            rz
            );

        world_vertices[i] =
            {{
                rx,
                ry,
                position_z + rz
            }};

        min_world_z =
            std::min(
                min_world_z,
                world_vertices[i][2]
                );
    }

    // --------------------------------------------------------
    // Analytical overlap
    // --------------------------------------------------------

    if(min_world_z < 0.0)
    {
        out_overlap = -min_world_z;
    }
    else
    {
        out_overlap = 0.0;
    }

    // --------------------------------------------------------
    // Contact normal
    // --------------------------------------------------------

    static constexpr double TIE_TOL = 1e-9;

    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_z = 0.0;

    int n_tied = 0;

    for(const auto& vertex : world_vertices)
    {
        if(vertex[2] <= min_world_z + TIE_TOL)
        {
            sum_x += vertex[0];
            sum_y += vertex[1];
            sum_z += vertex[2];
            n_tied++;
        }
    }

    double px = 0.0;
    double py = 0.0;
    double pz = 0.0;

    if(n_tied > 0)
    {
        px = sum_x / static_cast<double>(n_tied);
        py = sum_y / static_cast<double>(n_tied);
        pz = sum_z / static_cast<double>(n_tied);
    }

    // Vector from cube centroid to deepest support point.
    const double dx = px;
    const double dy = py;
    const double dz = pz - position_z;

    const double norm =
        std::sqrt(
            dx * dx +
            dy * dy +
            dz * dz
            );

    if(norm < 1e-12)
    {
        out_nx = 0.0;
        out_ny = 0.0;
        out_nz = -1.0;
    }
    else
    {
        out_nx = dx / norm;
        out_ny = dy / norm;
        out_nz = dz / norm;
    }
}

// ============================================================
// MAIN
// ============================================================

int main()
{
    std::cout
        << "Processing data ..."
        << std::endl;

    // ========================================================
    // LOAD SCALER
    // ========================================================

    if(!load_scaler("scalers.dat"))
    {
        return -1;
    }

    std::cout
        << "Scaler loaded\n";

    // ========================================================
    // LOAD MODEL
    // ========================================================

    torch::jit::script::Module model;

    try
    {
        model =
            torch::jit::load("overlap_model.pt");
    }
    catch(const c10::Error& error)
    {
        std::cerr
            << "Error loading overlap_model.pt:\n"
            << error.what()
            << "\n";

        return -1;
    }

    model.eval();

    torch::NoGradGuard no_grad;

    std::cout
        << "Model loaded\n";

    // ========================================================
    // OPEN INPUT FILE
    // ========================================================

    std::ifstream infile("training_data.csv");

    if(!infile.is_open())
    {
        std::cerr
            << "Could not open training_data.csv\n";

        return -1;
    }

    // ========================================================
    // READ INPUT HEADER
    // ========================================================

    std::string header;

    if(!std::getline(infile, header))
    {
        std::cerr
            << "Could not read CSV header\n";

        return -1;
    }

    trim_string(header);

    if(header.empty())
    {
        std::cerr
            << "CSV header is empty\n";

        return -1;
    }

    std::vector<std::string> header_tokens =
        split(header);

    for(auto& token : header_tokens)
    {
        trim_string(token);
    }

    std::cout
        << "Input columns found: "
        << header_tokens.size()
        << "\n";

    for(size_t i = 0; i < header_tokens.size(); ++i)
    {
        std::cout
            << "  ["
            << i
            << "] "
            << header_tokens[i]
            << "\n";
    }

    // ========================================================
    // FIND INPUT COLUMN INDICES
    // ========================================================

    int idx_z       = -1;
    int idx_qw      = -1;
    int idx_qx      = -1;
    int idx_qy      = -1;
    int idx_qz      = -1;
    int idx_overlap = -1;

    int idx_nx = -1;
    int idx_ny = -1;
    int idx_nz = -1;

    for(size_t i = 0; i < header_tokens.size(); ++i)
    {
        if(header_tokens[i] == "position_z")
        {
            idx_z = static_cast<int>(i);
        }
        else if(header_tokens[i] == "Q_orientation.w()")
        {
            idx_qw = static_cast<int>(i);
        }
        else if(header_tokens[i] == "Q_orientation.x()")
        {
            idx_qx = static_cast<int>(i);
        }
        else if(header_tokens[i] == "Q_orientation.y()")
        {
            idx_qy = static_cast<int>(i);
        }
        else if(header_tokens[i] == "Q_orientation.z()")
        {
            idx_qz = static_cast<int>(i);
        }
        else if(header_tokens[i] == "overlap(m)")
        {
            idx_overlap = static_cast<int>(i);
        }
        else if(header_tokens[i] == "normal_x")
        {
            idx_nx = static_cast<int>(i);
        }
        else if(header_tokens[i] == "normal_y")
        {
            idx_ny = static_cast<int>(i);
        }
        else if(header_tokens[i] == "normal_z")
        {
            idx_nz = static_cast<int>(i);
        }
    }

    if(
        idx_z  < 0 ||
        idx_qw < 0 ||
        idx_qx < 0 ||
        idx_qy < 0 ||
        idx_qz < 0
        )
    {
        std::cerr
            << "Missing one or more required columns.\n"
            << "Required columns are:\n"
            << "  position_z\n"
            << "  Q_orientation.w()\n"
            << "  Q_orientation.x()\n"
            << "  Q_orientation.y()\n"
            << "  Q_orientation.z()\n";

        return -1;
    }

    if(idx_overlap < 0)
    {
        std::cerr
            << "Warning: column 'overlap(m)' was not found.\n"
            << "Overlap statistics against the input data "
            << "will be skipped.\n";
    }

    const bool has_input_normal =
        idx_nx >= 0 &&
        idx_ny >= 0 &&
        idx_nz >= 0;

    if(!has_input_normal)
    {
        std::cout
            << "Input normal columns were not found.\n"
            << "Angular error will not be calculated against "
            << "input normals.\n";
    }

    // ========================================================
    // READ ALL DATA ROWS
    // ========================================================
    //
    // Reading first allows the program to detect rows containing
    // more columns than the header.
    // ========================================================

    std::vector<
        std::vector<std::string>
        > all_rows;

    std::string line;

    while(std::getline(infile, line))
    {
        trim_string(line);

        if(line.empty())
        {
            continue;
        }

        std::vector<std::string> tokens =
            split(line);

        for(auto& token : tokens)
        {
            trim_string(token);
        }

        all_rows.push_back(tokens);
    }

    infile.close();

    if(all_rows.empty())
    {
        std::cerr
            << "No data rows found\n";

        return -1;
    }

    // Determine the largest row width.
    size_t max_row_columns =
        header_tokens.size();

    for(const auto& row : all_rows)
    {
        max_row_columns =
            std::max(
                max_row_columns,
                row.size()
                );
    }

    // If some data row contains extra fields that are not
    // represented in the input header, preserve them with names.
    while(header_tokens.size() < max_row_columns)
    {
        const size_t index =
            header_tokens.size();

        header_tokens.push_back(
            "extra_input_column_" +
            std::to_string(index)
            );
    }

    // Pad shorter rows.
    for(auto& row : all_rows)
    {
        while(row.size() < header_tokens.size())
        {
            row.push_back("");
        }
    }

    // ========================================================
    // OPEN OUTPUT FILE
    // ========================================================

    std::ofstream outfile("predictions.csv");

    if(!outfile.is_open())
    {
        std::cerr
            << "Could not create predictions.csv\n";

        return -1;
    }

    outfile
        << std::setprecision(17);

    // ========================================================
    // WRITE COMPLETE OUTPUT HEADER
    // ========================================================

    std::vector<std::string> output_header =
        header_tokens;

    output_header.push_back("pred_overlap");
    output_header.push_back("pred_nx");
    output_header.push_back("pred_ny");
    output_header.push_back("pred_nz");
    output_header.push_back("pred_time_us");
    output_header.push_back("exact_overlap");
    output_header.push_back("exact_nx");
    output_header.push_back("exact_ny");
    output_header.push_back("exact_nz");
    output_header.push_back("exact_time_us");
    output_header.push_back("angular_error_deg");

    outfile
        << join_csv(output_header)
        << "\n";

    // ========================================================
    // STATISTICS
    // ========================================================

    long long count = 0;
    long long skipped_rows = 0;

    long long n_true_overlap = 0;

    double mse_overlap = 0.0;
    double mae_overlap = 0.0;
    double max_error_overlap = 0.0;

    double sum_angle_deg = 0.0;
    long long n_angle = 0;

    double sum_pred_time_us = 0.0;
    double sum_exact_time_us = 0.0;

    // ========================================================
    // PROCESS DATASET
    // ========================================================

    for(const auto& tokens : all_rows)
    {
        const size_t required_index =
            static_cast<size_t>(
                std::max({
                    idx_z,
                    idx_qw,
                    idx_qx,
                    idx_qy,
                    idx_qz
                })
                );

        if(tokens.size() <= required_index)
        {
            std::cerr
                << "Skipping malformed row: expected at least "
                << required_index + 1
                << " columns, found "
                << tokens.size()
                << "\n";

            skipped_rows++;
            continue;
        }

        // ----------------------------------------------------
        // READ REQUIRED INPUT VALUES
        // ----------------------------------------------------

        double position_z_double;
        double qw_double;
        double qx_double;
        double qy_double;
        double qz_double;

        const bool valid_required_values =
            try_parse_double(
                tokens[idx_z],
                position_z_double
                ) &&
            try_parse_double(
                tokens[idx_qw],
                qw_double
                ) &&
            try_parse_double(
                tokens[idx_qx],
                qx_double
                ) &&
            try_parse_double(
                tokens[idx_qy],
                qy_double
                ) &&
            try_parse_double(
                tokens[idx_qz],
                qz_double
                );

        if(!valid_required_values)
        {
            std::cerr
                << "Skipping row with invalid position or "
                << "quaternion values\n";

            skipped_rows++;
            continue;
        }

        float position_z =
            static_cast<float>(position_z_double);

        float qw =
            static_cast<float>(qw_double);

        float qx =
            static_cast<float>(qx_double);

        float qy =
            static_cast<float>(qy_double);

        float qz =
            static_cast<float>(qz_double);

        // ----------------------------------------------------
        // QUATERNION CANONICALIZATION
        // ----------------------------------------------------

        if(qw < 0.0f)
        {
            qw = -qw;
            qx = -qx;
            qy = -qy;
            qz = -qz;
        }

        // ====================================================
        // SURROGATE PREDICTION
        // ====================================================

        double pred_overlap = 0.0;
        double pred_nx = 0.0;
        double pred_ny = 0.0;
        double pred_nz = 0.0;

        const auto t_pred_start =
            std::chrono::steady_clock::now();

        torch::Tensor input =
            torch::tensor(
                {{
                    position_z,
                    qw,
                    qx,
                    qy,
                    qz
                }},
                torch::kFloat32
                );

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);

        torch::Tensor output =
            model.forward(inputs).toTensor();

        if(output.numel() < 4)
        {
            std::cerr
                << "Model output contains fewer than four "
                << "values. Expected "
                << "[log_overlap, nx, ny, nz].\n";

            return -1;
        }

        const float log_overlap_scaled =
            output[0][0].item<float>();

        const float nx_scaled =
            output[0][1].item<float>();

        const float ny_scaled =
            output[0][2].item<float>();

        const float nz_scaled =
            output[0][3].item<float>();

        // Inverse StandardScaler transformation:
        //
        // raw = scaled * scale + mean
        //

        const double log_overlap =
            static_cast<double>(
                log_overlap_scaled
                ) * y_scale[0] + y_mean[0];

        const double nx_raw =
            static_cast<double>(nx_scaled)
                * y_scale[1]
            + y_mean[1];

        const double ny_raw =
            static_cast<double>(ny_scaled)
                * y_scale[2]
            + y_mean[2];

        const double nz_raw =
            static_cast<double>(nz_scaled)
                * y_scale[3]
            + y_mean[3];

        pred_overlap =
            std::pow(10.0, log_overlap) - eps;

        if(
            !std::isfinite(pred_overlap) ||
            pred_overlap < 0.0
            )
        {
            pred_overlap = 0.0;
        }

        const double predicted_normal_norm =
            std::sqrt(
                nx_raw * nx_raw +
                ny_raw * ny_raw +
                nz_raw * nz_raw
                );

        if(
            predicted_normal_norm > 1e-12 &&
            std::isfinite(predicted_normal_norm)
            )
        {
            pred_nx =
                nx_raw / predicted_normal_norm;

            pred_ny =
                ny_raw / predicted_normal_norm;

            pred_nz =
                nz_raw / predicted_normal_norm;
        }
        else
        {
            pred_nx = 0.0;
            pred_ny = 0.0;
            pred_nz = -1.0;
        }

        const auto t_pred_end =
            std::chrono::steady_clock::now();

        const double pred_time_us =
            std::chrono::duration<double, std::micro>(
                t_pred_end - t_pred_start
                ).count();

        // ====================================================
        // EXACT ANALYTICAL CALCULATION
        // ====================================================

        double exact_overlap = 0.0;
        double exact_nx = 0.0;
        double exact_ny = 0.0;
        double exact_nz = 0.0;

        const auto t_exact_start =
            std::chrono::steady_clock::now();

        compute_exact_overlap_and_normal(
            static_cast<double>(qw),
            static_cast<double>(qx),
            static_cast<double>(qy),
            static_cast<double>(qz),
            static_cast<double>(position_z),
            exact_overlap,
            exact_nx,
            exact_ny,
            exact_nz
            );

        const auto t_exact_end =
            std::chrono::steady_clock::now();

        const double exact_time_us =
            std::chrono::duration<double, std::micro>(
                t_exact_end - t_exact_start
                ).count();

        sum_pred_time_us += pred_time_us;
        sum_exact_time_us += exact_time_us;

        // ====================================================
        // OVERLAP ACCURACY AGAINST INPUT overlap(m)
        // ====================================================

        double true_overlap = 0.0;
        bool have_true_overlap = false;

        if(
            idx_overlap >= 0 &&
            static_cast<size_t>(idx_overlap) < tokens.size()
            )
        {
            have_true_overlap =
                try_parse_double(
                    tokens[idx_overlap],
                    true_overlap
                    );
        }

        if(have_true_overlap)
        {
            const double error =
                pred_overlap - true_overlap;

            mse_overlap += error * error;
            mae_overlap += std::abs(error);

            max_error_overlap =
                std::max(
                    max_error_overlap,
                    std::abs(error)
                    );

            n_true_overlap++;
        }

        // ====================================================
        // NORMAL ANGULAR ERROR AGAINST INPUT NORMAL
        // ====================================================

        double angle_deg =
            std::numeric_limits<double>::quiet_NaN();

        if(
            has_input_normal &&
            static_cast<size_t>(idx_nz) < tokens.size()
            )
        {
            double true_nx;
            double true_ny;
            double true_nz;

            const bool valid_normal =
                try_parse_double(
                    tokens[idx_nx],
                    true_nx
                    ) &&
                try_parse_double(
                    tokens[idx_ny],
                    true_ny
                    ) &&
                try_parse_double(
                    tokens[idx_nz],
                    true_nz
                    );

            if(valid_normal)
            {
                const double true_normal_norm =
                    std::sqrt(
                        true_nx * true_nx +
                        true_ny * true_ny +
                        true_nz * true_nz
                        );

                if(true_normal_norm > 1e-12)
                {
                    true_nx /= true_normal_norm;
                    true_ny /= true_normal_norm;
                    true_nz /= true_normal_norm;

                    double cos_sim =
                        pred_nx * true_nx +
                        pred_ny * true_ny +
                        pred_nz * true_nz;

                    cos_sim =
                        std::max(
                            -1.0,
                            std::min(1.0, cos_sim)
                            );

                    angle_deg =
                        std::acos(cos_sim)
                        * 180.0
                        / PI;

                    sum_angle_deg += angle_deg;
                    n_angle++;
                }
            }
        }

        // ====================================================
        // WRITE OUTPUT ROW
        // ====================================================
        //
        // The input fields are written from tokens, not from
        // the original raw line. Therefore each row has the
        // same comma-separated structure as the header.
        // ====================================================

        outfile
            << join_csv(tokens)
            << ","
            << pred_overlap
            << ","
            << pred_nx
            << ","
            << pred_ny
            << ","
            << pred_nz
            << ","
            << pred_time_us
            << ","
            << exact_overlap
            << ","
            << exact_nx
            << ","
            << exact_ny
            << ","
            << exact_nz
            << ","
            << exact_time_us
            << ","
            << angle_deg
            << "\n";

        count++;
    }

    outfile.close();

    // ========================================================
    // FINAL STATISTICS
    // ========================================================

    std::cout
        << "\n================================================\n"
        << "INFERENCE RESULTS\n"
        << "================================================\n"
        << "Samples processed      : "
        << count
        << "\n"
        << "Rows skipped           : "
        << skipped_rows
        << "\n";

    if(n_true_overlap > 0)
    {
        const double mean_squared_error =
            mse_overlap /
            static_cast<double>(n_true_overlap);

        const double rmse_overlap =
            std::sqrt(mean_squared_error);

        const double mean_absolute_error =
            mae_overlap /
            static_cast<double>(n_true_overlap);

        std::cout
            << "Rows with overlap      : "
            << n_true_overlap
            << "\n"
            << "Overlap RMSE           : "
            << rmse_overlap
            << "\n"
            << "Overlap MAE            : "
            << mean_absolute_error
            << "\n"
            << "Overlap Max error      : "
            << max_error_overlap
            << "\n";
    }
    else
    {
        std::cout
            << "No valid input overlap values found.\n";
    }

    if(n_angle > 0)
    {
        const double mean_angle_deg =
            sum_angle_deg /
            static_cast<double>(n_angle);

        std::cout
            << "Mean normal angle error: "
            << mean_angle_deg
            << " deg (n="
            << n_angle
            << ")\n";
    }
    else
    {
        std::cout
            << "No valid input normal values found.\n";
    }

    if(count > 0)
    {
        const double avg_pred_time_us =
            sum_pred_time_us /
            static_cast<double>(count);

        const double avg_exact_time_us =
            sum_exact_time_us /
            static_cast<double>(count);

        const double speedup =
            sum_pred_time_us > 0.0
                ? sum_exact_time_us / sum_pred_time_us
                : 0.0;

        std::cout
            << "\n================================================\n"
            << "TIMING RESULTS\n"
            << "================================================\n"
            << "Surrogate  - sum : "
            << sum_pred_time_us
            << " us | avg : "
            << avg_pred_time_us
            << " us\n"
            << "Analytical - sum : "
            << sum_exact_time_us
            << " us | avg : "
            << avg_exact_time_us
            << " us\n"
            << "Speed-up (analytical/surrogate) : "
            << speedup
            << "x\n";
    }

    std::cout
        << "\nPredictions saved to predictions.csv\n";

    return 0;
}
