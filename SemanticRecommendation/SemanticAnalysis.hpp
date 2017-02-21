#pragma once
#include "Import.hpp"
#include "DataSet.hpp"
#include "DataModel.hpp"
#include "Model.hpp"
#include "Utils.hpp"
#include "Solver.hpp"

class RecommendationSemanticAnalysisUnit
{
public:
	const DataModel& dm;

protected:
	const int dim;
	const double reg_factor;
	const AdaDelta& solver;

protected:
	struct  
	{
		vector<vector<vec>>	user;
		vector<vector<vec>>	item;
		vector<mat>	trans_join;
	} _p, _derv_grad, _derv_x;

public:
	RecommendationSemanticAnalysisUnit
		(const DataModel& dm, int dim, double reg_factor, const AdaDelta& solver)
		:dm(dm), dim(dim), reg_factor(reg_factor), solver(solver)
	{
		_p.user.resize(dm.n_rate);
		for_each(_p.user.begin(), _p.user.end(), [&](vector<vec>& elem){init_vector_vec(elem, dm.n_user, dim); });
		_p.item.resize(dm.n_rate);
		for_each(_p.item.begin(), _p.item.end(), [&](vector<vec>& elem){init_vector_vec(elem, dm.n_item, dim); });
		init_vector_mat(_p.trans_join, dm.n_rate, dim);

		_derv_grad.user.resize(dm.n_rate);
		for_each(_derv_grad.user.begin(), _derv_grad.user.end(), [&](vector<vec>& elem){zeros_vector_vec(elem, dm.n_user, dim); });
		_derv_grad.item.resize(dm.n_rate);
		for_each(_derv_grad.item.begin(), _derv_grad.item.end(), [&](vector<vec>& elem){zeros_vector_vec(elem, dm.n_item, dim); });
		zeros_vector_mat(_derv_grad.trans_join, dm.n_rate, dim);

		_derv_x = _derv_grad;

		//init_vector_vec(_p.user, dm.n_user, dim);
		//init_vector_vec(_p.item, dm.n_item, dim);
		//init_vector_mat(_p.trans_join, dm.n_rate, dim);

		//zeros_vector_vec(_derv_grad.user, dm.n_user, dim);
		//zeros_vector_vec(_derv_grad.item, dm.n_item, dim);
		//zeros_vector_mat(_derv_grad.trans_join, dm.n_rate, dim);

		//zeros_vector_vec(_derv_x.user, dm.n_user, dim);
		//zeros_vector_vec(_derv_x.item, dm.n_item, dim);
		//zeros_vector_mat(_derv_x.trans_join, dm.n_rate, dim);
	}

public:
	virtual double probability(const int user, const int item, const int rating)
	{
		return probability(make_tuple(user, item, rating));
	}

	virtual double probability(const tuple<int, int, int>& datum)
	{
		vec& v_user = _p.user[get<2>(datum)][get<0>(datum)];
		vec& v_item = _p.item[get<2>(datum)][get<1>(datum)];
		mat& m_trans = _p.trans_join[get<2>(datum)];

		return  as_scalar(v_user.t() * m_trans * v_item) + mini_factor;
	}

	virtual void train_once(const int id_user, const int id_item, const int id_rating, const double exfactor = 1.0)
	{
		vec& v_user = _p.user[id_rating][id_user];
		vec& v_item = _p.item[id_rating][id_item];
		mat& m_trans = _p.trans_join[id_rating];

		vec grad_user = exfactor * m_trans* v_item - reg_factor * v_user;
		vec grad_item = exfactor * m_trans.t() * v_user - reg_factor * v_item;
		mat grad_trans = exfactor * v_item * v_user.t() - reg_factor * m_trans;

		solver.gradient_ascent_positive(_derv_grad.user[id_rating][id_user], _derv_x.user[id_rating][id_user], 
			v_user, grad_user);
		solver.gradient_ascent_positive(_derv_grad.item[id_rating][id_item], _derv_x.item[id_rating][id_item], 
			v_item, grad_item);
		solver.gradient_ascent_positive(_derv_grad.trans_join[id_rating], _derv_x.trans_join[id_rating], 
			m_trans, grad_trans);
	}

public:
	virtual void save(FormatFile & file)
	{
		file << _p.user << _p.item << _p.trans_join;
		file << _derv_grad.user << _derv_grad.item << _derv_grad.trans_join;
		file << _derv_x.user << _derv_x.item << _derv_x.trans_join;
	}

	virtual  void load(FormatFile & file)
	{
		file >> _p.user >> _p.item >> _p.trans_join;
		file >> _derv_grad.user >> _derv_grad.item >> _derv_grad.trans_join;
		file >> _derv_x.user >> _derv_x.item >> _derv_x.trans_join;
	}

public:
	int infer_user_category(int user, const vec& prior) const
	{
		return infer_user_distribution(user, prior).index_max();
	}

	int infer_item_category(int item, const vec& prior) const
	{
		return infer_item_distribution(item, prior).index_max();
	}

	vec infer_user_distribution(int user, const vec& prior) const
	{
		vec res = zeros(dim);
		vec pri = zeros(dim);

		for (auto r = 1; r < dm.n_rate; ++r)
		{
			for (auto i = 1; i < dm.n_item; ++i)
			{
				pri = _p.trans_join[r] * _p.item[r][i];
			}
		}

		for (auto r = 1; r < dm.n_rate; ++r)
		{
			res += _p.user[r][user] * prior[r];
		}
		return res % pri / accu(res % pri);
	}

	vec infer_item_distribution(int item, const vec& prior) const
	{
		vec res = zeros(dim);
		vec pri = zeros(dim);

		for (auto r = 1; r < dm.n_rate; ++r)
		{
			for (auto i = 1; i < dm.n_user; ++i)
			{
				pri = _p.trans_join[r].t() * _p.user[r][i];
			}
		}

		for (auto r = 1; r < dm.n_rate; ++r)
		{
			res += _p.item[r][item] * prior[r];
		}
		return res % pri / accu(res % pri);
	}
};

class RecommendationSemanticAnalysis
	:public Model
{
protected:
	const int dim;
	const double reg_factor;
	const AdaDelta& solver;

protected:
	struct
	{
		vector<vec> user_bias;
		vector<vec> item_bias;
	} _p, _derv_grad, _derv_x;
	int n_unit;

protected:
	vector<RecommendationSemanticAnalysisUnit*> units;

public:
	RecommendationSemanticAnalysis
		(const DataModel& ds, int n_unit, int dim, double reg_factor,  const AdaDelta& solver)
		:Model(ds), dim(dim), solver(solver), n_unit(n_unit), reg_factor(reg_factor)
	{
		units.resize(1000);
		units.resize(0);

		logout.record() << "[Variant]\tKSA.1.1 Alpha";
		logout.record() << "[Setting]\tUnit Number = " << n_unit;
		logout.record() << "[Setting]\tDim Number = " << dim;
		logout.record() << "[Setting]\tRegularization Factor = " << reg_factor;
		solver.logout_parameter();

		for (auto i = 0; i < n_unit; ++i)
		{
			units.push_back(new RecommendationSemanticAnalysisUnit(dm, dim, reg_factor, solver));
		}

		init_vector_vec(_p.user_bias, dm.n_user, dm.n_rate);
		init_vector_vec(_p.item_bias, dm.n_item, dm.n_rate);

		zeros_vector_vec(_derv_grad.user_bias, dm.n_user, dm.n_rate);
		zeros_vector_vec(_derv_grad.item_bias, dm.n_item, dm.n_rate);

		zeros_vector_vec(_derv_x.user_bias, dm.n_user, dm.n_rate);
		zeros_vector_vec(_derv_x.item_bias, dm.n_item, dm.n_rate);
	}

public:
	virtual double probability(const tuple<int, int, int>& datum)
	{
		vec prob_unit(units.size());
		for (auto u = units.begin(); u != units.end(); ++u)
		{
			prob_unit[u - units.begin()] = (*u)->probability(datum);
		}

		return sum(prob_unit)
			* _p.user_bias[get<0>(datum)][get<2>(datum)]
			* _p.item_bias[get<1>(datum)][get<2>(datum)] + mini_factor;
	}

	virtual void train_core(const tuple<int, int, int>& datum, const double factor, const double fractial, const double partial)
	{
		const int user = get<0>(datum);
		const int item = get<1>(datum);
		const int rate = get<2>(datum);

		vec grad_user_bias = zeros(dm.n_rate);
		vec grad_item_bias = zeros(dm.n_rate);

		for (auto neg_rate = 1; neg_rate < dm.n_rate; ++neg_rate)
		{
			double ex_grad = factor * (neg_rate - fractial / partial) / partial;

			for (auto u = units.begin(); u != units.end(); ++u)
			{
				double grad_negcur = probability_simple(user, item, neg_rate);

				(*u)->train_once(user, item, neg_rate, ex_grad *
					_p.user_bias[user][neg_rate] * _p.item_bias[item][neg_rate]);

				grad_user_bias[neg_rate] += ex_grad * grad_negcur / (grad_user_bias[neg_rate] + mini_factor)
					-  reg_factor * _p.user_bias[user][neg_rate];
				grad_item_bias[neg_rate] += ex_grad * grad_negcur / (grad_item_bias[neg_rate] + mini_factor)
					- reg_factor * _p.item_bias[item][neg_rate];

			}
		}

		solver.gradient_ascent_positive
			(_derv_grad.user_bias[user], _derv_x.user_bias[user], _p.user_bias[user], grad_user_bias);
		solver.gradient_ascent_positive
			(_derv_grad.item_bias[item], _derv_x.item_bias[item], _p.item_bias[item], grad_item_bias);
	}

	virtual void train_once(const tuple<int, int, int>& datum)
	{
		const int user = get<0>(datum);
		const int item = get<1>(datum);
		const int rate = get<2>(datum);

		double fractial = 0;
		double partial = 0;
		for (auto neg_rate = 1; neg_rate < dm.n_rate; ++neg_rate)
		{
			double prob_tuple = probability_simple(user, item, neg_rate);

			fractial += neg_rate * prob_tuple;
			partial += prob_tuple;
		}

		double error = fractial / partial - get<2>(datum);
		double factor = -error;

		train_core(datum, factor, fractial, partial);
	}

public:
	int get_unit_number() const
	{
		return n_unit;
	}

public:
	virtual double infer(const int user, const int item) override
	{
		double fractial = 0;
		double partial = 0;
		for (auto ind_r = 1; ind_r < dm.n_rate; ++ind_r)
		{
			double prob_tuple = probability(make_tuple(user, item, ind_r));
			fractial += ind_r * prob_tuple;
			partial += prob_tuple;
		}

		return fractial / partial;
	}

	public:
		int infer_user_category(int user, int id_unit) const
		{
			return units[id_unit]->infer_user_category(user, _p.user_bias[user]);
		}

		int infer_item_category(int item, int id_unit) const
		{
			return units[id_unit]->infer_item_category(item, _p.item_bias[item]);
		}

		vec infer_user_distribution(int user, int id_unit) const
		{
			return units[id_unit]->infer_user_distribution(user, _p.user_bias[user]);
		}

		vec infer_item_distribution(int item, int id_unit) const
		{
			return units[id_unit]->infer_item_distribution(item, _p.item_bias[item]);
		}

public:
	virtual void save(FormatFile & file) override
	{
		file << n_unit;
		file << _p.user_bias << _p.item_bias;
		file << _derv_grad.user_bias << _derv_grad.item_bias;
		file << _derv_x.user_bias << _derv_x.item_bias;

		for (auto u = units.begin(); u != units.end(); ++u)
		{
			(*u)->save(file);
		}
	}

	virtual void load(FormatFile & file) override
	{
		file >> n_unit;
		file >> _p.user_bias >> _p.item_bias;
		file >> _derv_grad.user_bias >> _derv_grad.item_bias;
		file >> _derv_x.user_bias >> _derv_x.item_bias;

		for (auto u = units.begin(); u != units.end(); ++u)
		{
			delete (*u);
		}

		units.resize(n_unit);
		for (auto u = units.begin(); u != units.end(); ++u)
		{
			(*u) = new RecommendationSemanticAnalysisUnit(dm, dim, reg_factor, solver);
			(*u)->load(file);
		}
	}
};

class RSAUnitLaplace
	:public RecommendationSemanticAnalysisUnit
{
protected:
	const double sigma;

public:
	RSAUnitLaplace
		(const DataModel& dm, int dim, double reg_factor, double sigma, const AdaDelta& Solver)
		:RecommendationSemanticAnalysisUnit(dm, dim, reg_factor, Solver), sigma(sigma)
	{
		;
	}

public:
	virtual double probability(const tuple<int, int, int>& datum)
	{
		vec& v_user = _p.user[get<2>(datum)][get<0>(datum)];
		vec& v_item = _p.item[get<2>(datum)][get<1>(datum)];
		mat& m_trans = _p.trans_join[get<2>(datum)];

		return  as_scalar(v_user.t() * m_trans * v_item) * exp(-sum(abs(v_user - v_item)) / sigma) + mini_factor;
	}

	virtual void train_once(const int id_user, const int id_item, const int id_rating, const double exfactor = 1.0)
	{
		vec& v_user = _p.user[id_rating][id_user];
		vec& v_item = _p.item[id_rating][id_item];
		mat& m_trans = _p.trans_join[id_rating];

		double back = exp(-sum(abs(v_user - v_item)) / sigma);
		double front = as_scalar(v_user.t() * m_trans * v_item);

		vec grad_user = exfactor * m_trans* v_item * back - exfactor * front * back * sign(v_user - v_item) / sigma
			- reg_factor * (v_user);
		vec grad_item = exfactor * m_trans.t() * v_user  * back + exfactor * front * back * sign(v_user - v_item) / sigma
			- reg_factor * (v_item);
		mat grad_trans = exfactor * v_item * v_user.t() * back - reg_factor * (m_trans);

		solver.gradient_ascent_positive(_derv_grad.user[id_rating][id_user], _derv_x.user[id_rating][id_user],
			v_user, grad_user);
		solver.gradient_ascent_positive(_derv_grad.item[id_rating][id_item], _derv_x.item[id_rating][id_item],
			v_item, grad_item);
		solver.gradient_ascent_positive(_derv_grad.trans_join[id_rating], _derv_x.trans_join[id_rating],
			m_trans, grad_trans);
	}

public:
	vec infer_user_distribution(int user, const vec& prior) const
	{
		vec res = zeros(dim);
		vec pri = zeros(dim);

		for (auto r = 1; r < dm.n_rate; ++r)
		{
			for (auto i = 1; i < dm.n_item; ++i)
			{
				pri = _p.trans_join[r] * _p.item[r][i] * exp(-sum(abs(_p.user[r][user] - _p.item[r][i])));
			}
		}

		for (auto r = 1; r < dm.n_rate; ++r)
		{
			res += _p.user[r][user] * prior[r];
		}
		return res % pri / accu(res % pri);
	}

	vec infer_item_distribution(int item, const vec& prior) const
	{
		vec res = zeros(dim);
		vec pri = zeros(dim);

		for (auto r = 1; r < dm.n_rate; ++r)
		{
			for (auto i = 1; i < dm.n_user; ++i)
			{
				pri = _p.trans_join[r].t() * _p.user[r][i] * exp(-sum(abs(_p.item[r][item] - _p.user[r][i])));
			}
		}

		for (auto r = 1; r < dm.n_rate; ++r)
		{
			res += _p.item[r][item] * prior[r];
		}
		return res % pri / accu(res % pri);
	}
};

class RSALaplace
	: public RecommendationSemanticAnalysis
{
protected:
	const double sigma;
	const double prior_bias;

public:
	RSALaplace
		(const DataModel& dm, int n_unit, int dim, double reg_factor, double sigma, double prior_bias, const AdaDelta& solver)
		: RecommendationSemanticAnalysis(dm, n_unit, dim, reg_factor, solver), sigma(sigma), prior_bias(prior_bias)
	{
		logout.record() << "[Variant]\tKSA.1.3 Laplace";
		logout.record() << "[Setting]\tSigma = " << sigma;
		logout.record() << "[Setting]\tPrior Bias = " << prior_bias;
		
		for (auto u = units.begin(); u != units.end(); ++u)
		{
			delete (*u);
		}
		units.clear();

		for (auto i = 0; i < n_unit; ++i)
		{
			units.push_back(new RSAUnitLaplace(dm, dim, reg_factor, sigma, solver));
		}
	}

public:
	virtual double probability(const tuple<int, int, int>& datum)
	{
		vec prob_unit(units.size());
		for (auto u = units.begin(); u != units.end(); ++u)
		{
			prob_unit[u - units.begin()] = (*u)->probability(datum);
		}

		return exp(sum(prob_unit)
			* (prior_bias +_p.user_bias[get<0>(datum)][get<2>(datum)])
			* (prior_bias+_p.item_bias[get<1>(datum)][get<2>(datum)]));
	}

	virtual void train_core(const tuple<int, int, int>& datum, const double factor, const double fractial, const double partial)
	{
		const int user = get<0>(datum);
		const int item = get<1>(datum);
		const int rate = get<2>(datum);

		vec grad_user_bias = zeros(dm.n_rate);
		vec grad_item_bias = zeros(dm.n_rate);

		for (auto neg_rate = 1; neg_rate < dm.n_rate; ++neg_rate)
		{
			double ex_grad = factor * (neg_rate - fractial / partial) / partial;

			for (auto u = units.begin(); u != units.end(); ++u)
			{
				double grad_negcur = probability_simple(user, item, neg_rate);

				(*u)->train_once(user, item, neg_rate, ex_grad * grad_negcur 
					* (prior_bias +_p.user_bias[user][neg_rate]) * (prior_bias +_p.item_bias[item][neg_rate]));

				grad_user_bias[neg_rate] += ex_grad * grad_negcur * log(grad_negcur)
					/ (grad_user_bias[neg_rate] + prior_bias)
					-  reg_factor * sign(_p.user_bias[user][neg_rate]);
				grad_item_bias[neg_rate] += ex_grad * grad_negcur * log(grad_negcur)
					/ (grad_item_bias[neg_rate] + prior_bias)
					-  reg_factor * sign(_p.item_bias[item][neg_rate]);
			}
		}

		solver.gradient_ascent_positive
			(_derv_grad.user_bias[user], _derv_x.user_bias[user], _p.user_bias[user], grad_user_bias);
		solver.gradient_ascent_positive
			(_derv_grad.item_bias[item], _derv_x.item_bias[item], _p.item_bias[item], grad_item_bias);
	}

public:
	virtual void load(FormatFile & file) override
	{
		file >> n_unit;
		file >> _p.user_bias >> _p.item_bias;
		file >> _derv_grad.user_bias >> _derv_grad.item_bias;
		file >> _derv_x.user_bias >> _derv_x.item_bias;

		for (auto u = units.begin(); u != units.end(); ++u)
		{
			delete (*u);
		}

		units.resize(n_unit);
		for (auto u = units.begin(); u != units.end(); ++u)
		{
			(*u) = new RSAUnitLaplace(dm, dim, reg_factor, sigma, solver);
			(*u)->load(file);
		}
	}
};

class RSALaplaceCRP
	: public RSALaplace
{
protected:
	const double prior_CRP;
	const int max_unit;

protected:
	int n_cur_unit;

public:
	RSALaplaceCRP
		(const DataModel& ds, int n_unit, int dim, double reg_factor, double sigma, double prior_bias, 
		double prior_CRP, const AdaDelta& solver)
		: RSALaplace(ds, n_unit, dim, reg_factor, sigma, prior_bias, solver),
		prior_CRP(prior_CRP), max_unit(max_unit), n_cur_unit(n_unit)
	{
		logout.record() << "[Variant]\tKSA.1.6 Laplace CRP";
		logout.record() << "[Setting]\tPrior CRP = " << prior_CRP;
	}

public:
	virtual void train_kernel()
	{
		RSALaplace::train_kernel();

		double acc_componet = 0;
		double acc_ubn = 0;

#pragma omp parallel for
		for (auto i = dm.ds_train.begin(); i != dm.ds_train.end(); ++i)
		{
			for (auto u = units.begin(); u != units.end(); ++u)
			{
				acc_componet += exp((*u)->probability(*i) *
					(prior_bias + _p.user_bias[get<0>(*i)][get<2>(*i)] * _p.item_bias[get<1>(*i)][get<2>(*i)]));
			}

			acc_ubn += pow(exp(prior_bias + _p.user_bias[get<0>(*i)][get<2>(*i)] * _p.item_bias[get<1>(*i)][get<2>(*i)])
				/ exp(1), - n_cur_unit) / probability(*i);
		}

#pragma omp critical
		{
			acc_componet /= dm.ds_train.size();
			acc_ubn /= dm.ds_train.size();
			if (prior_CRP * acc_ubn / (prior_CRP * acc_ubn + acc_componet) > randu())
			{
				units.push_back(new RSAUnitLaplace(dm, dim, reg_factor, sigma, solver));
				++n_cur_unit;

				logout.record() << "[Info]\tADD a New Component (Total " << n_cur_unit
					<< ", Cap " << units.capacity() << ") with probability " << prior_CRP * acc_ubn / (prior_CRP * acc_ubn + acc_componet) 
					<< " and current likelihood = "<< acc_componet;
			}
			else
			{
				logout.record() << "[Info]\tSamping Probability " << prior_CRP * acc_ubn / (prior_CRP * acc_ubn + acc_componet)
					<< " and current likelihood = " << acc_componet;
			}
		}
	}
};

class RecSA
	:public RSALaplace
{
public:
	RecSA
		(const DataModel& ds, int n_unit, int dim, double reg_factor, double sigma, double prior_bias,
		const AdaDelta& solver)
		:RSALaplace(ds, n_unit, dim, reg_factor, sigma, prior_bias, solver)
	{
		logout.record() << "[Variant]\tRecSA 2.0";
	}

public:
	virtual double probability(const tuple<int, int, int>& datum)
	{
		vec prob_unit(units.size());
		for (auto u = units.begin(); u != units.end(); ++u)
		{
			prob_unit[u - units.begin()] = (*u)->probability(datum);
		}

		return exp(sum(prob_unit)
			 + _p.user_bias[get<0>(datum)][get<2>(datum)]
			 + _p.item_bias[get<1>(datum)][get<2>(datum)]);
	}

	virtual void train_core(const tuple<int, int, int>& datum, const double factor, const double fractial, const double partial)
	{
		const int user = get<0>(datum);
		const int item = get<1>(datum);
		const int rate = get<2>(datum);

		vec grad_user_bias = zeros(dm.n_rate);
		vec grad_item_bias = zeros(dm.n_rate);

		for (auto neg_rate = 1; neg_rate < dm.n_rate; ++neg_rate)
		{
			double ex_grad = factor * (neg_rate - fractial / partial) / partial;

			for (auto u = units.begin(); u != units.end(); ++u)
			{
				double grad_negcur = probability_simple(user, item, neg_rate);

				(*u)->train_once(user, item, neg_rate, ex_grad * grad_negcur);

				grad_user_bias[neg_rate] += ex_grad * grad_negcur
					- reg_factor * sign(_p.user_bias[user][neg_rate]);
				grad_item_bias[neg_rate] += ex_grad * grad_negcur
					- reg_factor * sign(_p.item_bias[item][neg_rate]);
			}
		}

		solver.gradient_ascent_positive
			(_derv_grad.user_bias[user], _derv_x.user_bias[user], _p.user_bias[user], grad_user_bias);
		solver.gradient_ascent_positive
			(_derv_grad.item_bias[item], _derv_x.item_bias[item], _p.item_bias[item], grad_item_bias);
	}
};