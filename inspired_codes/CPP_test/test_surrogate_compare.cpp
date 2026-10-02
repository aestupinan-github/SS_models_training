// ============================================================
// GINN TORCHSCRIPT INFERENCE TEST
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
// FEATURES:
//   - Matches Python preprocessing
//   - Quaternion canonicalization
//   - Contact classification
//   - Overlap recovery (surrogate)
//   - Exact analytical cube-plane overlap (vertex rotation + min-z)
//   - Per-row wall-clock timing for BOTH methods
//   - Runtime statistics (sum / average / speed-up)
//
// ============================================================
#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <array>

// ============================================================
// GLOBAL SCALER DATA
// ============================================================
std::vector<double> x_mean;
std::vector<double> x_scale;
double y_mean = 0.0;
double y_scale = 1.0;
double overlap_eps = 1e-15;

// ============================================================
// GEOMETRY CONSTANTS (must match dataset generator)
// ============================================================
static constexpr double CUBE_SIZE = 1.0;
static constexpr double HALF      = CUBE_SIZE / 2.0;

// ============================================================
// CSV SPLIT
// ============================================================
std::vector<std::string> split(const std::string& line)
{
    std::stringstream ss(line);
    std::string item;
    std::vector<std::string> tokens;
    while(std::getline(ss, item, ','))
    {
        tokens.push_back(item);
    }
    return tokens;
}

// ============================================================
// LOAD SCALERS
// ============================================================

bool load_scaler(const std::string& filename)
{
    std::ifstream file(filename);

    if(!file.is_open())
    {
        std::cerr << "Could not open scaler file\n";

        return false;
    }

    std::string line;

    while(std::getline(file, line))
    {
        if(line == "x_mean")
        {
            std::getline(file, line);
            std::stringstream ss(line);
            double val;
            while(ss >> val)
            {
                x_mean.push_back(val);
            }
        }
        if(line == "x_scale")
        {
            std::getline(file, line);
            std::stringstream ss(line);
            double val;
            while(ss >> val)
            {
                x_scale.push_back(val);
            }
        }
        if(line == "y_mean")
        {
            std::getline(file, line);
            y_mean = std::stod(line);
        }
        if(line == "y_scale")
        {
            std::getline(file, line);
            y_scale = std::stod(line);
        }
        if(line == "eps")
        {
            std::getline(file, line);

            overlap_eps = std::stod(line);
        }
    }

    file.close();
    if(
        x_mean.size() != 5
        ||
        x_scale.size() != 5
        )
    {
        std::cerr
            << "Invalid scaler data\n";
        return false;
    }
    return true;
}

// ============================================================
// SIGMOID
// ============================================================
double sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

// ============================================================
// QUATERNION ROTATION OF A 3D POINT
// ------------------------------------------------------------
// Standard active rotation: p' = q * p * q_conj
// Quaternion convention here: (qw, qx, qy, qz), matches the
// dataset / model input convention.
// ============================================================
inline void rotate_point_by_quat(
    double qw, double qx, double qy, double qz,
    double px, double py, double pz,
    double& out_x, double& out_y, double& out_z
    )
{
    // Rotation matrix built directly from quaternion components
    // (avoids quaternion-quaternion multiplication overhead;
    // exactly equivalent to p' = q p q^-1 for unit quaternions)
    const double R00 = 1.0 - 2.0 * (qy*qy + qz*qz);
    const double R01 = 2.0 * (qx*qy - qz*qw);
    const double R02 = 2.0 * (qx*qz + qy*qw);

    const double R10 = 2.0 * (qx*qy + qz*qw);
    const double R11 = 1.0 - 2.0 * (qx*qx + qz*qz);
    const double R12 = 2.0 * (qy*qz - qx*qw);

    const double R20 = 2.0 * (qx*qz - qy*qw);
    const double R21 = 2.0 * (qy*qz + qx*qw);
    const double R22 = 1.0 - 2.0 * (qx*qx + qy*qy);

    out_x = R00*px + R01*py + R02*pz;
    out_y = R10*px + R11*py + R12*pz;
    out_z = R20*px + R21*py + R22*pz;
}

// ============================================================
// EXACT ANALYTICAL CUBE-PLANE OVERLAP
// ------------------------------------------------------------
// Builds the 8 cube vertices in the local (body) frame at
// (+-HALF, +-HALF, +-HALF), rotates each by the input quaternion,
// translates by position_z, and takes the minimum resulting
// world z. If that minimum is negative, the cube penetrates the
// plane (z=0) by exactly that amount -> overlap = -min_z.
// Otherwise overlap = 0 (no contact / clearance case).
//
// This is the same "generate vertices -> rotate -> find minima"
// geometric definition used earlier in this project as the
// ground-truth check on the trained surrogate.
// ============================================================
inline double compute_exact_overlap(
    double qw, double qx, double qy, double qz,
    double position_z,
    bool& is_contact
    )
{
    static const std::array<std::array<double,3>, 8> local_vertices = {{
        {-HALF, -HALF, -HALF},
        {-HALF, -HALF,  HALF},
        {-HALF,  HALF, -HALF},
        {-HALF,  HALF,  HALF},
        { HALF, -HALF, -HALF},
        { HALF, -HALF,  HALF},
        { HALF,  HALF, -HALF},
        { HALF,  HALF,  HALF}
    }};

    double min_world_z = std::numeric_limits<double>::infinity();

    for(const auto& v : local_vertices)
    {
        double rx, ry, rz;
        rotate_point_by_quat(
            qw, qx, qy, qz,
            v[0], v[1], v[2],
            rx, ry, rz
            );
        const double world_z = position_z + rz;
        if(world_z < min_world_z)
        {
            min_world_z = world_z;
        }
    }

    if(min_world_z < 0.0)
    {
        is_contact = true;
        return -min_world_z;
    }
    else
    {
        is_contact = false;
        return 0.0;
    }
}

// ============================================================
// MAIN
// ============================================================
int main()
{
    std::cout << "Processing data ..." << std::endl;

    // ========================================================
    // LOAD SCALERS
    // ========================================================
    if(!load_scaler("scalers.dat"))
    {
        return -1;
    }
    std::cout << "Scalers loaded\n";

    // ========================================================
    // LOAD MODEL
    // ========================================================
    torch::jit::script::Module model;
    try
    {
        model = torch::jit::load("overlap_model.pt");
    }
    catch(const c10::Error& e)
    {
        std::cerr << "Error loading model\n";
        std::cerr << e.what() << "\n";
        return -1;
    }
    model.eval();
    torch::NoGradGuard no_grad;   // avoid autograd bookkeeping in the loop
    std::cout << "Model loaded\n";

    // ========================================================
    // INPUT CSV
    // ========================================================
    std::ifstream infile("training_data.csv");

    if(!infile.is_open())
    {
        std::cerr << "Could not open training_data.csv\n";

        return -1;
    }

    // ========================================================
    // OUTPUT CSV
    // ========================================================
    std::ofstream outfile("predictions.csv");

    if(!outfile.is_open())
    {
        std::cerr << "Could not create predictions.csv\n";

        return -1;
    }

    // ========================================================
    // READ HEADER
    // ========================================================
    std::string header;

    std::getline(infile, header);

    outfile
        << header
        << ",contact_prob"
        << ",pred_contact"
        << ",pred_overlap"
        << ",pred_time_us"
        << ",exact_overlap"
        << ",exact_contact"
        << ",exact_time_us\n";

    // ========================================================
    // FIND COLUMN INDICES
    // ========================================================
    auto header_tokens = split(header);

    int idx_z  = -1;
    int idx_qw = -1;
    int idx_qx = -1;
    int idx_qy = -1;
    int idx_qz = -1;
    int idx_overlap = -1;
    for(size_t i = 0; i < header_tokens.size(); ++i)
    {
        if(header_tokens[i] == "position_z")           idx_z = i;
        if(header_tokens[i] == "Q_orientation.w()")     idx_qw = i;
        if(header_tokens[i] == "Q_orientation.x()")     idx_qx = i;
        if(header_tokens[i] == "Q_orientation.y()")     idx_qy = i;
        if(header_tokens[i] == "Q_orientation.z()")     idx_qz = i;
        if(header_tokens[i] == "overlap(m)")            idx_overlap = i;
    }
    if(idx_z < 0 || idx_qw < 0 || idx_qx < 0 || idx_qy < 0 || idx_qz < 0)
    {
        std::cerr << "Missing required columns\n";

        return -1;
    }

    // ========================================================
    // STATS
    // ========================================================
    long long count = 0;
    long long correct_contact = 0;
    double mse = 0.0;
    double mae = 0.0;
    double max_error = 0.0;

    double sum_pred_time_us  = 0.0;
    double sum_exact_time_us = 0.0;

    // ========================================================
    // LOOP OVER DATASET
    // ========================================================
    std::string line;
    while(std::getline(infile, line))
    {
        auto tokens = split(line);
        if(tokens.size() <= static_cast<size_t>(std::max({idx_z, idx_qw, idx_qx, idx_qy, idx_qz})))
        {
            std::cerr << "Skipping malformed line\n";
            continue;
        }

        // ----------------------------------------------------
        // READ INPUTS
        // ----------------------------------------------------
        float position_z = std::stof(tokens[idx_z]);
        float qw =         std::stof(tokens[idx_qw]);
        float qx =         std::stof(tokens[idx_qx]);
        float qy =         std::stof(tokens[idx_qy]);
        float qz =         std::stof(tokens[idx_qz]);

        // ----------------------------------------------------
        // QUATERNION CANONICALIZATION (model input only)
        // ----------------------------------------------------
        float model_qw = qw, model_qx = qx, model_qy = qy, model_qz = qz;
        if(model_qw < 0.0f)
        {
            model_qw = -model_qw;
            model_qx = -model_qx;
            model_qy = -model_qy;
            model_qz = -model_qz;
        }

        // ----------------------------------------------------
        // NORMALIZATION
        // ----------------------------------------------------
        float input_z  = (position_z - x_mean[0]) / x_scale[0];
        float input_qw = (model_qw   - x_mean[1]) / x_scale[1];
        float input_qx = (model_qx   - x_mean[2]) / x_scale[2];
        float input_qy = (model_qy   - x_mean[3]) / x_scale[3];
        float input_qz = (model_qz   - x_mean[4]) / x_scale[4];

        // ======================================================
        // SURROGATE PREDICTION — TIMED (tensor build + forward)
        // ======================================================
        double contact_prob   = 0.0;
        bool   pred_contact   = false;
        double pred_overlap   = 0.0;

        auto t_pred_start = std::chrono::steady_clock::now();

        torch::Tensor input = torch::tensor(
            {{ input_z, input_qw, input_qx, input_qy, input_qz }},
            torch::kFloat32
            );

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        torch::Tensor output = model.forward(inputs).toTensor();

        float contact_logit       = output[0][0].item<float>();
        float overlap_log_scaled  = output[0][1].item<float>();

        contact_prob = sigmoid(contact_logit);
        pred_contact = contact_prob > 0.5;

        if(pred_contact)
        {
            double overlap_log = overlap_log_scaled * y_scale + y_mean;
            pred_overlap = std::pow(10.0, overlap_log) - overlap_eps;
            if(pred_overlap < 0.0) pred_overlap = 0.0;
        }

        auto t_pred_end = std::chrono::steady_clock::now();
        double pred_time_us =
            std::chrono::duration<double, std::micro>(t_pred_end - t_pred_start).count();

        // ======================================================
        // EXACT ANALYTICAL CALCULATION — TIMED
        // ======================================================
        bool exact_contact = false;
        double exact_overlap = 0.0;

        auto t_exact_start = std::chrono::steady_clock::now();

        exact_overlap = compute_exact_overlap(
            static_cast<double>(qw), static_cast<double>(qx),
            static_cast<double>(qy), static_cast<double>(qz),
            static_cast<double>(position_z),
            exact_contact
            );

        auto t_exact_end = std::chrono::steady_clock::now();
        double exact_time_us =
            std::chrono::duration<double, std::micro>(t_exact_end - t_exact_start).count();

        sum_pred_time_us  += pred_time_us;
        sum_exact_time_us += exact_time_us;

        // ----------------------------------------------------
        // TRUE OVERLAP (from dataset, for accuracy stats)
        // ----------------------------------------------------
        double true_overlap = 0.0;
        if(idx_overlap >= 0)
        {
            true_overlap = std::stod(tokens[idx_overlap]);
        }
        bool true_contact = true_overlap > 0.0;

        if(pred_contact == true_contact)
        {
            correct_contact++;
        }
        double err = pred_overlap - true_overlap;
        mse += err * err;
        mae += std::abs(err);
        max_error = std::max(max_error, std::abs(err));

        // ----------------------------------------------------
        // WRITE CSV (row + surrogate + exact + timings)
        // ----------------------------------------------------

        outfile
            << line << ","
            << contact_prob << ","
            << pred_contact << ","
            << pred_overlap << ","
            << pred_time_us << ","
            << exact_overlap << ","
            << exact_contact << ","
            << exact_time_us << "\n";

        count++;
    }
    infile.close();
    outfile.close();

    // ========================================================
    // FINAL STATS
    // ========================================================
    mse /= static_cast<double>(count);
    mae /= static_cast<double>(count);
    double rmse = std::sqrt(mse);
    double contact_accuracy =
        100.0 * static_cast<double>(correct_contact) / static_cast<double>(count);

    double avg_pred_time_us  = sum_pred_time_us  / static_cast<double>(count);
    double avg_exact_time_us = sum_exact_time_us / static_cast<double>(count);
    double speedup = (sum_pred_time_us > 0.0) ? (sum_exact_time_us / sum_pred_time_us) : 0.0;

    std::cout << "\n================================================\n";
    std::cout << "INFERENCE RESULTS\n";
    std::cout << "================================================\n";
    std::cout << "Samples processed : " << count << "\n";
    std::cout << "Contact accuracy  : " << contact_accuracy << " %\n";
    std::cout << "RMSE              : " << rmse << "\n";
    std::cout << "MAE               : " << mae << "\n";
    std::cout << "Max error         : " << max_error << "\n";

    std::cout << "\n================================================\n";
    std::cout << "TIMING RESULTS (prediction/calculation only)\n";
    std::cout << "================================================\n";
    std::cout << "Surrogate  - sum : " << sum_pred_time_us  << " us | avg : " << avg_pred_time_us  << " us\n";
    std::cout << "Analytical - sum : " << sum_exact_time_us << " us | avg : " << avg_exact_time_us << " us\n";
    std::cout << "Speed-up (analytical/surrogate) : " << speedup << "x\n";

    std::cout << "\nPredictions saved to predictions.csv\n";
    return 0;
}
