#ifndef __BROWNIAN_TREE_HPP__
#define __BROWNIAN_TREE_HPP__

#include <cmath>
#include <map>
#include <memory>
#include <random>
#include <vector>

#include "ggml.h"
#include "rng.hpp"

/**
 * BrownianTree - Lazy binary tree for temporally-correlated Brownian motion sampling
 *
 * This class generates Brownian motion noise that maintains temporal consistency
 * across arbitrary time queries, unlike independent noise sampling. This is critical
 * for SDE samplers like DPM++ 3M SDE where noise correlation affects fine detail quality.
 *
 * Algorithm: Binary subdivision with Brownian bridge interpolation
 * - Values are computed lazily on first query and cached
 * - Given W(s) and W(u), intermediate W(t) is sampled from:
 *   mean = ((u-t)*W(s) + (t-s)*W(u)) / (u-s)
 *   variance = (u-t)*(t-s) / (u-s)
 *
 * Reference: https://github.com/google-research/torchsde (BrownianTree)
 */
class BrownianTree {
private:
    float t_min_;
    float t_max_;
    uint64_t seed_;

    // Cache for computed Brownian values at specific times
    // Key: time value, Value: W(t)
    std::map<float, float> cache_;

    // RNG for generating new random values
    std::mt19937_64 gen_;
    std::normal_distribution<float> normal_dist_;

    // Tolerance for floating point comparison
    static constexpr float EPSILON = 1e-6f;

    // Check if two floats are approximately equal
    bool approx_equal(float a, float b) const {
        return std::fabs(a - b) < EPSILON;
    }

    // Find cached value at time t, or return nullptr if not found
    const float* find_cached(float t) const {
        for (const auto& kv : cache_) {
            if (approx_equal(kv.first, t)) {
                return &kv.second;
            }
        }
        return nullptr;
    }

    // Get or compute W(t) using binary subdivision
    float get_w(float t) {
        // Check bounds
        if (t <= t_min_) return 0.0f;
        if (t >= t_max_) {
            // Return cached endpoint or compute it
            const float* cached = find_cached(t_max_);
            if (cached) return *cached;
            // This shouldn't happen if properly initialized
            return cache_[t_max_];
        }

        // Check cache first
        const float* cached = find_cached(t);
        if (cached) return *cached;

        // Find bracketing cached values using binary search approach
        float left_t = t_min_;
        float left_w = 0.0f;
        float right_t = t_max_;
        float right_w = cache_[t_max_];

        // Find tightest brackets from cache
        for (const auto& kv : cache_) {
            if (kv.first < t && kv.first > left_t) {
                left_t = kv.first;
                left_w = kv.second;
            }
            if (kv.first > t && kv.first < right_t) {
                right_t = kv.first;
                right_w = kv.second;
            }
        }

        // Recursively subdivide until we reach t
        return subdivide(left_t, left_w, right_t, right_w, t);
    }

    // Subdivide interval [s, u] to compute W(t) using Brownian bridge
    float subdivide(float s, float w_s, float u, float w_u, float t) {
        // If interval is small enough, interpolate directly
        float mid = (s + u) / 2.0f;

        // Compute or retrieve midpoint
        const float* cached_mid = find_cached(mid);
        float w_mid;

        if (cached_mid) {
            w_mid = *cached_mid;
        } else {
            // Brownian bridge interpolation for midpoint
            // W(mid) | W(s), W(u) ~ N(mean, variance)
            // mean = (W(s) + W(u)) / 2  (since mid is exactly halfway)
            // variance = (u - s) / 4
            float mean = (w_s + w_u) / 2.0f;
            float std_dev = std::sqrt((u - s) / 4.0f);
            w_mid = mean + std_dev * normal_dist_(gen_);
            cache_[mid] = w_mid;
        }

        // Check if t is close to midpoint
        if (approx_equal(t, mid)) {
            return w_mid;
        }

        // Recurse into appropriate half
        if (t < mid) {
            return subdivide(s, w_s, mid, w_mid, t);
        } else {
            return subdivide(mid, w_mid, u, w_u, t);
        }
    }

public:
    /**
     * Construct a BrownianTree for the time interval [t_min, t_max]
     *
     * @param t_min Minimum time (typically -log(sigma_max))
     * @param t_max Maximum time (typically -log(sigma_min))
     * @param seed Random seed for reproducibility
     */
    BrownianTree(float t_min, float t_max, uint64_t seed)
        : t_min_(t_min), t_max_(t_max), seed_(seed),
          gen_(seed), normal_dist_(0.0f, 1.0f) {

        // Initialize endpoint: W(t_min) = 0, W(t_max) = sqrt(t_max - t_min) * Z
        cache_[t_min_] = 0.0f;
        float dt = t_max_ - t_min_;
        cache_[t_max_] = std::sqrt(dt) * normal_dist_(gen_);
    }

    /**
     * Sample Brownian increment W(t1) - W(t0), normalized by sqrt(|t1 - t0|)
     * This matches the interface expected by k-diffusion SDE samplers
     *
     * @param t0 Start time
     * @param t1 End time
     * @return Normalized Brownian increment
     */
    float operator()(float t0, float t1) {
        float w0 = get_w(t0);
        float w1 = get_w(t1);
        float dt = std::fabs(t1 - t0);
        if (dt < EPSILON) return 0.0f;
        return (w1 - w0) / std::sqrt(dt);
    }

    /**
     * Reset the tree with a new seed
     */
    void reset(uint64_t seed) {
        seed_ = seed;
        gen_.seed(seed);
        cache_.clear();
        cache_[t_min_] = 0.0f;
        float dt = t_max_ - t_min_;
        cache_[t_max_] = std::sqrt(dt) * normal_dist_(gen_);
    }

    /**
     * Get current cache size (for debugging)
     */
    size_t cache_size() const {
        return cache_.size();
    }
};

/**
 * BrownianTreeNoiseSampler - Tensor-level Brownian noise sampler
 *
 * Wraps BrownianTree to provide correlated noise for entire tensors,
 * maintaining per-element Brownian paths for consistent fine details.
 */
class BrownianTreeNoiseSampler {
private:
    std::vector<BrownianTree> trees_;  // One tree per tensor element
    size_t num_elements_;
    float t_min_;
    float t_max_;

public:
    /**
     * Construct sampler for a tensor of given size
     *
     * @param num_elements Number of elements in the tensor
     * @param sigma_min Minimum sigma value
     * @param sigma_max Maximum sigma value
     * @param seed Base random seed
     */
    BrownianTreeNoiseSampler(size_t num_elements, float sigma_min, float sigma_max, uint64_t seed)
        : num_elements_(num_elements) {

        // Transform sigma to log-space time (like k-diffusion)
        t_min_ = -std::log(sigma_max);
        t_max_ = -std::log(sigma_min);

        // Create one Brownian tree per element with different seeds
        trees_.reserve(num_elements);
        for (size_t i = 0; i < num_elements; i++) {
            // Use seed mixing to get independent but reproducible per-element seeds
            uint64_t elem_seed = seed ^ (static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ULL);
            trees_.emplace_back(t_min_, t_max_, elem_seed);
        }
    }

    /**
     * Sample Brownian noise increment for transition from sigma to sigma_next
     * Fills the output array with correlated noise values
     *
     * @param sigma Current sigma value
     * @param sigma_next Next sigma value
     * @param output Output array (must be size num_elements_)
     */
    void sample(float sigma, float sigma_next, float* output) {
        float t0 = -std::log(sigma);
        float t1 = -std::log(sigma_next);

        for (size_t i = 0; i < num_elements_; i++) {
            output[i] = trees_[i](t0, t1);
        }
    }

    /**
     * Sample directly into a ggml tensor
     */
    void sample_tensor(float sigma, float sigma_next, struct ggml_tensor* tensor) {
        if (static_cast<size_t>(ggml_nelements(tensor)) != num_elements_) {
            // Size mismatch - this shouldn't happen
            return;
        }
        float* data = (float*)tensor->data;
        sample(sigma, sigma_next, data);
    }

    /**
     * Reset all trees with a new base seed
     */
    void reset(uint64_t seed) {
        for (size_t i = 0; i < num_elements_; i++) {
            uint64_t elem_seed = seed ^ (static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ULL);
            trees_[i].reset(elem_seed);
        }
    }

    size_t num_elements() const { return num_elements_; }
};

#endif  // __BROWNIAN_TREE_HPP__
