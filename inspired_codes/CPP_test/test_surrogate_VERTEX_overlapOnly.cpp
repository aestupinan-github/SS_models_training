// ============================================================
// GINN TORCHSCRIPT INFERENCE TEST — SCALAR OVERLAP-ONLY MODEL
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
// The input CSV may contain additional columns. All input columns
// are preserved in the output. The following columns are appended:
//
//   pred_overlap
//   pred_time_us
//   exact_overlap
//   exact_time_us
//
// Required input columns:
//
//   position_z
//   Q_orientation.w()
//   Q_orientation.x()
//   Q_orientation.y()
//   Q_orientation.z()
//
// Optional input column:
//
//   overlap(m)
//
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
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================
// GLOBAL SCALER DATA
// ============================================================

double y_mean  = 0.0;
double y_scale = 1.0;
double eps     = 1e-15;

// ============================================================
// GEOMETRY CONSTANTS
// ============================================================

static constexpr double CUBE_SIZE = 1.0;
static constexpr double HALF      = CUBE_SIZE / 2.0;

// ============================================================
// SPLIT CSV LINE
// ------------------------------------------------------------
// This is a simple comma-separated parser. It assumes that the
// CSV fields themselves do not contain quoted commas.
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

    // Preserve a final empty field if the line ends with a comma.
    if(!line.empty() && line.back() == ',')
    {
        tokens.push_back("");
    }

    return tokens;
}

// ============================================================
// JOIN CSV TOKENS
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
// REMOVE CARRIAGE RETURN AND TRIM WHITESPACE
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
    while(!value.empty() &&
           std::isspace(
               static_cast<unsigned char>(value.front())
               ))
    {
        value.erase(value.begin());
    }

    // Remove trailing whitespace.
    while(!value.empty() &&
           std::isspace(
               static_cast<unsigned char>(value.back())
               ))
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

        // Reject strings such as "1.2abc".
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

    std::string line;
    bool found_y_mean  = false;
    bool found_y_scale = false;

    while(std::getline(file, line))
    {
        trim_string(line);

        if(line == "y_mean")
        {
            if(!std::getline(file, line))
            {
                std::cerr << "Missing value after y_mean\n";
                return false;
            }

            trim_string(line);

            try
            {
                y_mean = std::stod(line);
                found_y_mean = true;
            }
            catch(const std::exception&)
            {
                std::cerr << "Invalid y_mean value\n";
                return false;
            }
        }
        else if(line == "y_scale")
        {
            if(!std::getline(file, line))
            {
                std::cerr << "Missing value after y_scale\n";
                return false;
            }

            trim_string(line);

            try
            {
                y_scale = std::stod(line);
                found_y_scale = true;
            }
            catch(const std::exception&)
            {
                std::cerr << "Invalid y_scale value\n";
                return false;
            }
        }
        else if(line == "eps")
        {
            if(!std::getline(file, line))
            {
                std::cerr << "Missing value after eps\n";
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

    if(!found_y_mean || !found_y_scale)
    {
        std::cerr
            << "Invalid scaler data. Expected y_mean and y_scale.\n";

        return false;
    }

    if(y_scale == 0.0)
    {
        std::cerr
            << "Invalid scaler data: y_scale is zero.\n";

        return false;
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
    const double R00 = 1.0 - 2.0 * (qy * qy + qz * qz);
    const double R01 = 2.0 * (qx * qy - qz * qw);
    const double R02 = 2.0 * (qx * qz + qy * qw);

    const double R10 = 2.0 * (qx * qy + qz * qw);
    const double R11 = 1.0 - 2.0 * (qx * qx + qz * qz);
    const double R12 = 2.0 * (qy * qz - qx * qw);

    const double R20 = 2.0 * (qx * qz - qy * qw);
    const double R21 = 2.0 * (qy * qz + qx * qw);
    const double R22 = 1.0 - 2.0 * (qx * qx + qy * qy);

    out_x = R00 * px + R01 * py + R02 * pz;
    out_y = R10 * px + R11 * py + R12 * pz;
    out_z = R20 * px + R21 * py + R22 * pz;
}

// ============================================================
// EXACT ANALYTICAL CUBE-PLANE OVERLAP
// ============================================================

inline double compute_exact_overlap(
    double qw,
    double qx,
    double qy,
    double qz,
    double position_z
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

    double min_world_z =
        std::numeric_limits<double>::infinity();

    for(const auto& vertex : local_vertices)
    {
        double rx;
        double ry;
        double rz;

        rotate_point_by_quat(
            qw,
            qx,
            qy,
            qz,
            vertex[0],
            vertex[1],
            vertex[2],
            rx,
            ry,
            rz
            );

        const double world_z = position_z + rz;

        if(world_z < min_world_z)
        {
            min_world_z = world_z;
        }
    }

    return min_world_z < 0.0
               ? -min_world_z
               : 0.0;
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
        << "Scaler loaded\n"
        << "  y_mean  = " << y_mean << "\n"
        << "  y_scale = " << y_scale << "\n"
        << "  eps     = " << eps << "\n";

    // ========================================================
    // LOAD MODEL
    // ========================================================

    torch::jit::script::Module model;

    try
    {
        model = torch::jit::load("overlap_model.pt");
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
    // OPEN INPUT CSV
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
            << "Could not read the input CSV header\n";

        return -1;
    }

    trim_string(header);

    if(header.empty())
    {
        std::cerr
            << "The input CSV header is empty\n";

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
            << "  [" << i << "] "
            << header_tokens[i]
            << "\n";
    }

    // ========================================================
    // FIND REQUIRED COLUMN INDICES
    // ========================================================

    int idx_z       = -1;
    int idx_qw      = -1;
    int idx_qx      = -1;
    int idx_qy      = -1;
    int idx_qz      = -1;
    int idx_overlap = -1;

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
            << "Warning: optional column 'overlap(m)' "
            << "was not found.\n"
            << "Accuracy statistics against the input overlap "
            << "will be skipped.\n";
    }

    // ========================================================
    // READ ALL INPUT DATA ROWS FIRST
    // --------------------------------------------------------
    // This allows us to determine whether some rows contain
    // additional columns before writing the output header.
    // ========================================================

    std::vector<std::vector<std::string>> all_rows;
    std::string line;

    while(std::getline(infile, line))
    {
        trim_string(line);

        // Skip empty physical lines.
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
            << "No data rows found in training_data.csv\n";

        return -1;
    }

    // Determine the maximum number of fields in any data row.
    size_t max_row_columns = header_tokens.size();

    for(const auto& row : all_rows)
    {
        max_row_columns =
            std::max(max_row_columns, row.size());
    }

    // If some data rows have additional fields that were not
    // represented in the header, give those fields generic names.
    while(header_tokens.size() < max_row_columns)
    {
        const size_t new_index = header_tokens.size();

        header_tokens.push_back(
            "extra_input_column_" +
            std::to_string(new_index)
            );
    }

    // Pad shorter rows so that all rows have the same number
    // of fields as the final input header.
    for(auto& row : all_rows)
    {
        while(row.size() < header_tokens.size())
        {
            row.push_back("");
        }
    }

    // ========================================================
    // OPEN OUTPUT CSV
    // ========================================================

    std::ofstream outfile("predictions.csv");

    if(!outfile.is_open())
    {
        std::cerr
            << "Could not create predictions.csv\n";

        return -1;
    }

    outfile << std::setprecision(17);

    // Write one complete output header line.
    std::vector<std::string> output_header =
        header_tokens;

    output_header.push_back("pred_overlap");
    output_header.push_back("pred_time_us");
    output_header.push_back("exact_overlap");
    output_header.push_back("exact_time_us");

    outfile
        << join_csv(output_header)
        << "\n";

    // ========================================================
    // STATS
    // ========================================================

    long long count = 0;
    long long count_with_true_overlap = 0;

    double mse_overlap       = 0.0;
    double mae_overlap       = 0.0;
    double max_error_overlap = 0.0;

    double sum_pred_time_us  = 0.0;
    double sum_exact_time_us = 0.0;

    long long skipped_rows = 0;

    // ========================================================
    // PROCESS DATASET
    // ========================================================

    for(const auto& input_tokens : all_rows)
    {
        const auto& tokens = input_tokens;

        // Check required fields.
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
                << "Skipping malformed row: not enough columns\n";

            skipped_rows++;
            continue;
        }

        // ----------------------------------------------------
        // READ INPUT VALUES
        // ----------------------------------------------------

        double position_z_double;
        double qw_double;
        double qx_double;
        double qy_double;
        double qz_double;

        if(
            !try_parse_double(
                tokens[idx_z],
                position_z_double
                ) ||
            !try_parse_double(
                tokens[idx_qw],
                qw_double
                ) ||
            !try_parse_double(
                tokens[idx_qx],
                qx_double
                ) ||
            !try_parse_double(
                tokens[idx_qy],
                qy_double
                ) ||
            !try_parse_double(
                tokens[idx_qz],
                qz_double
                )
            )
        {
            std::cerr
                << "Skipping row with invalid position or "
                << "quaternion value\n";

            skipped_rows++;
            continue;
        }

        float position_z = static_cast<float>(
            position_z_double
            );

        float qw = static_cast<float>(qw_double);
        float qx = static_cast<float>(qx_double);
        float qy = static_cast<float>(qy_double);
        float qz = static_cast<float>(qz_double);

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
        // SURROGATE PREDICTION — TIMED
        // ====================================================

        double pred_overlap = 0.0;

        const auto t_pred_start =
            std::chrono::steady_clock::now();

        torch::Tensor input = torch::tensor(
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

        // Expected model output shape: [1, 1].
        float log_overlap_scaled =
            output[0][0].item<float>();

        double log_overlap =
            static_cast<double>(
                log_overlap_scaled
                ) * y_scale + y_mean;

        pred_overlap =
            std::pow(10.0, log_overlap) - eps;

        if(pred_overlap < 0.0)
        {
            pred_overlap = 0.0;
        }

        const auto t_pred_end =
            std::chrono::steady_clock::now();

        const double pred_time_us =
            std::chrono::duration<double, std::micro>(
                t_pred_end - t_pred_start
                ).count();

        // ====================================================
        // EXACT ANALYTICAL CALCULATION — TIMED
        // ====================================================

        const auto t_exact_start =
            std::chrono::steady_clock::now();

        const double exact_overlap =
            compute_exact_overlap(
                static_cast<double>(qw),
                static_cast<double>(qx),
                static_cast<double>(qy),
                static_cast<double>(qz),
                static_cast<double>(position_z)
                );

        const auto t_exact_end =
            std::chrono::steady_clock::now();

        const double exact_time_us =
            std::chrono::duration<double, std::micro>(
                t_exact_end - t_exact_start
                ).count();

        sum_pred_time_us  += pred_time_us;
        sum_exact_time_us += exact_time_us;

        // ====================================================
        // ACCURACY STATISTICS AGAINST INPUT OVERLAP
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

            count_with_true_overlap++;
        }

        // ====================================================
        // WRITE OUTPUT ROW
        // ----------------------------------------------------
        // Write all original input fields, followed by the
        // generated output fields. No original line is copied
        // blindly.
        // ====================================================

        outfile
            << join_csv(tokens)
            << ","
            << pred_overlap
            << ","
            << pred_time_us
            << ","
            << exact_overlap
            << ","
            << exact_time_us
            << "\n";

        count++;
    }

    outfile.close();

    // ========================================================
    // FINAL STATS
    // ========================================================

    std::cout
        << "\n================================================\n"
        << "INFERENCE RESULTS\n"
        << "================================================\n"
        << "Samples processed  : "
        << count
        << "\n"
        << "Rows skipped       : "
        << skipped_rows
        << "\n";

    if(count_with_true_overlap > 0)
    {
        mse_overlap /=
            static_cast<double>(
                count_with_true_overlap
                );

        mae_overlap /=
            static_cast<double>(
                count_with_true_overlap
                );

        const double rmse_overlap =
            std::sqrt(mse_overlap);

        std::cout
            << "Rows with overlap  : "
            << count_with_true_overlap
            << "\n"
            << "Overlap RMSE       : "
            << rmse_overlap
            << "\n"
            << "Overlap MAE        : "
            << mae_overlap
            << "\n"
            << "Overlap Max error  : "
            << max_error_overlap
            << "\n";
    }
    else
    {
        std::cout
            << "No valid 'overlap(m)' values were found.\n"
            << "Overlap accuracy statistics were not calculated.\n";
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
