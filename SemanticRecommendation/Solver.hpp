#pragma once
#include "Import.hpp"
#include "Logging.hpp"

class AdaDelta
{
protected:
	const double moment;
	const double regularization;

public:
	AdaDelta(double moment, double regularization)
		:moment(moment), regularization(regularization)
	{
		;
	}

public:
	void logout_parameter() const
	{
		logout.record() << "[Solver]\tAdaDelta";
		logout.record() << "[Setting]\tMoment = " << moment;
		logout.record() << "[Setting]\tRegularization = " << regularization;
	}

public:
	void gradient_ascent_positive(double& derv_grad, double& derv_x, double& elem, double& grad) const
	{
		derv_grad = moment * derv_grad + (1 - moment) * grad * grad;
		double derv_elem = sqrt(derv_x + regularization) / sqrt(derv_grad + regularization) * grad;
		derv_x = moment * derv_x + (1 - moment) * derv_elem * derv_elem;

		elem += derv_elem;
		elem = fabs(elem);
		elem = min(1.0, elem);
	}

	void gradient_ascent_positive(vec& derv_grad, vec& derv_x, vec& elem, vec& grad) const
	{
		derv_grad = moment * derv_grad + (1 - moment) * grad % grad;
		vec derv_elem = sqrt(derv_x + regularization) / sqrt(derv_grad + regularization) % grad;
		derv_x = moment * derv_x + (1 - moment) * derv_elem % derv_elem;

#pragma omp critical
		{
			elem += derv_elem;
			elem %= sign(elem);
			elem /= accu(elem);
		}
	}

	void gradient_ascent_positive(mat& derv_grad, mat& derv_x, mat& elem, mat& grad) const
	{
		derv_grad = moment * derv_grad + (1 - moment) * grad % grad;
		mat derv_elem = sqrt(derv_x + regularization) / sqrt(derv_grad + regularization) % grad;
		derv_x = moment * derv_x + (1 - moment) * derv_elem % derv_elem;

#pragma omp critical
		{
			elem += derv_elem;
			elem %= sign(elem);
			elem /= accu(elem);
		}
	}
};