#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <vector>

namespace rfs
{

/**
 * Pose-only motion prior. With xi_j = Logmap(T_{j-1}^-1 T_j) the body-frame increment of frame j,
 * the error is xi_k - sum_i coeffs[i] * xi_{k-1-i}.
 * Constant velocity is {1}, constant acceleration {2,-1}, constant jerk {3,-3,1}, and constant
 * acceleration damped as exp(-t/tau) is {1+e,-e} with e = exp(-dt/tau).
 * Keys are ordered k-1, k, k-2, k-3, ... so the first two match a BetweenFactor(k-1, k).
 */
class MotionPriorFactor : public gtsam::NoiseModelFactor
{
public:
	MotionPriorFactor(int k, const std::vector<double> &coeffs, const gtsam::SharedNoiseModel &noise)
		: gtsam::NoiseModelFactor(noise, makeKeys(k, coeffs.size())), coeffs_(coeffs) {}

	gtsam::Vector unwhitenedError(const gtsam::Values &x, gtsam::OptionalMatrixVecType H = nullptr) const override
	{
		const int n = coeffs_.size();
		// poses in time order: P[0] = k-n-1 ... P[n+1] = k
		std::vector<gtsam::Pose3> P(n + 2);
		for (int j = 0; j < n + 2; j++)
			P[j] = x.at<gtsam::Pose3>(keys_[slot(j)]);

		gtsam::Vector6 err = gtsam::Vector6::Zero();
		if (H)
		{
			H->assign(n + 2, gtsam::Matrix::Zero(6, 6));
		}
		// increment j (1..n+1) goes from P[j-1] to P[j], weighted s
		for (int j = 1; j < n + 2; j++)
		{
			double s = (j == n + 1) ? 1.0 : -coeffs_[n - j];
			gtsam::Matrix6 Hb1, Hlog;
			gtsam::Pose3 b = P[j - 1].between(P[j], H ? &Hb1 : nullptr);
			err += s * gtsam::Pose3::Logmap(b, H ? &Hlog : nullptr);
			if (H)
			{
				(*H)[slot(j - 1)] += s * Hlog * Hb1;
				(*H)[slot(j)] += s * Hlog;
			}
		}
		return err;
	}

	gtsam::NonlinearFactor::shared_ptr clone() const override
	{
		return std::make_shared<MotionPriorFactor>(*this);
	}

private:
	std::vector<double> coeffs_;

	/** position in keys_ of the j-th pose in time order */
	int slot(int j) const
	{
		const int n = coeffs_.size();
		if (j == n + 1)
			return 1;
		if (j == n)
			return 0;
		return n + 1 - j;
	}

	static gtsam::KeyVector makeKeys(int k, int n)
	{
		gtsam::KeyVector keys = {gtsam::Key(k - 1), gtsam::Key(k)};
		for (int i = 0; i < n; i++)
			keys.push_back(k - 2 - i);
		return keys;
	}
};

} // namespace rfs
