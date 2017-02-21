#pragma once
#include "Import.hpp"
#include "DataSet.hpp"
#include "DataModel.hpp"
#include "File.hpp"
#include "Logging.hpp"
#include "Model.hpp"
#include <boost/progress.hpp>

class Task
{
protected:
	const DataModel& dm;

protected:
	double mean_bias;
	double bound_max;
	double bound_min;

public:
	Model* model;

public:
	Task(Model* model)
		:dm(model->dm), model(model)
	{
		;
	}

	Task(DataModel& dm)
		:dm(dm)
	{
		;
	}

	void set_model(Model* modelt)
	{
		this->model = modelt;
	}

public:
	virtual void test() = 0;
	virtual void result() = 0;
};

class TopNPerformanceTask
	:public Task
{
protected:
	const int N;

protected:
	struct Performance
	{
		double HR;
		double ARHR;
	} best, current;
public:
	TopNPerformanceTask(Model *model, int N = 10)
		:Task(model), N(N)
	{
		best.HR = 0;
		best.ARHR = 0;
	}

public:
	virtual void test() override
	{
		current.HR = 0;
		current.ARHR = 0;
		double total = 0;

		for (auto u = 0; u < dm.n_user; ++u)
		{
			if (dm.ds_test_hit[u].size() == 0)
				continue; 

			++total;

			vector<pair<double, int>> scores(dm.n_item);
			for (auto i = 1; i < dm.n_item; ++i)
			{
				if (dm.ds_train_hit[u].find(i) != dm.ds_train_hit[u].end())
					scores[i] = make_pair(0, i);
				else
					scores[i] =make_pair(model->infer(u, i), i);
			}

			sort(scores.begin(), scores.end(), greater<pair<double, int>>());
			for (auto r = 0; r < N; ++r)
			{
				if (dm.ds_test_hit[u].find(scores[r].second) != dm.ds_test_hit[u].end())
				{
					++current.HR;
					current.ARHR+= 1 / double(r + 1);
				}
			}
		}

		current.HR /= total;
		current.ARHR /= total;

		best.HR = max(best.HR, current.HR);
		best.ARHR = max(best.ARHR, current.ARHR);

		string inc_mark_hr = "";
		if (best.HR <= current.HR)
			inc_mark_hr = "[+]";
		else
			inc_mark_hr = " []";

		string inc_mark_arhr = "";
		if (best.ARHR <= current.ARHR)
			inc_mark_arhr = "[+]";
		else
			inc_mark_arhr = " []";

		logout.record() << "[Result]\tHR = " << current.HR / total
			<< ", Best-HR = " << best.HR / total << inc_mark_hr;
		logout.record() << "[Result]\tARHR = " << current.ARHR / total
			<< ", Best-ARHR = " << best.ARHR / total << inc_mark_arhr;
	}
};

class BasicPredictPerformanceTask
	:public Task
{
public:
	struct Performance
	{
		double bm_HIT;
		double bm_MAE;
		double bm_RMSE;
	} best, current;
	const bool open_stein_effect;

public:
	BasicPredictPerformanceTask(DataModel& dm, bool open_stein_effect = false)
		:Task(dm), open_stein_effect(open_stein_effect)
	{
		best.bm_HIT = 0;
		best.bm_MAE = 1e10;
		best.bm_RMSE = 1e10;
	}

	BasicPredictPerformanceTask(Model *model, bool open_stein_effect = false)
		:Task(model), open_stein_effect(open_stein_effect)
	{
		best.bm_HIT = 0;
		best.bm_MAE = 1e10;
		best.bm_RMSE = 1e10;
	}

public:
	virtual void test() override
	{
		mean_bias = 0;
		bound_max = 0;
		bound_min = 1e10;

		if (open_stein_effect)
		{
			for (auto i = dm.ds_train.begin(); i != dm.ds_train.end(); ++i)
			{
				bound_max = max(bound_max, model->infer(get<0>(*i), get<1>(*i)));
				bound_min = min(bound_min, model->infer(get<0>(*i), get<1>(*i)));
			}

			for (auto i = dm.ds_train.begin(); i != dm.ds_train.end(); ++i)
			{
				mean_bias += model->infer(get<0>(*i), get<1>(*i)) - get<2>(*i);
			}

			mean_bias /= dm.ds_train.size();

			logout.record() << "[Model Analysis]\tMax Bound = " << bound_max;
			logout.record() << "[Model Analysis]\tMin Bound = " << bound_min;
			logout.record() << "[Model Analysis]\tMean Bias= " << mean_bias;
		}

		logout.record() << "[Task]\tBasic Predict Performance Task";

		current.bm_HIT = 0;
		current.bm_MAE = 0;
		current.bm_RMSE = 0;

		int n_TOTAL = 0;
		for (auto i = dm.ds_test.begin(); i != dm.ds_test.end(); ++i)
		{
			++ n_TOTAL;
			double pred_rating = model->infer(get<0>(*i), get<1>(*i)) -mean_bias;

			if (int(pred_rating + 0.5) == get<2>(*i))
				++ current.bm_HIT;

			current.bm_MAE += abs(model->discrete_infer(get<0>(*i), get<1>(*i)) - get<2>(*i));
			current.bm_RMSE += (pred_rating - get<2>(*i)) * ( pred_rating - get<2>(*i));
		}

		string inc_mark_hit = "";
		if (best.bm_HIT <= current.bm_HIT)
			inc_mark_hit = "[+]";
		else
			inc_mark_hit = "";

		string inc_mark_mae = "";
		if (best.bm_MAE >= current.bm_MAE)
			inc_mark_mae = "[+]";
		else
			inc_mark_mae = "";

		string inc_mark_rmse = "";
		if (best.bm_RMSE >= current.bm_RMSE)
			inc_mark_rmse = "[+]";
		else
			inc_mark_rmse = "";

		best.bm_HIT = max(best.bm_HIT, current.bm_HIT);
		best.bm_MAE = min(best.bm_MAE, current.bm_MAE);
		best.bm_RMSE = min(best.bm_RMSE, current.bm_RMSE);

		logout.record() << "[Result]\tHIT-Ratio = " << current.bm_HIT / n_TOTAL
			<< ", Best-HIT-Ratio = " << best.bm_HIT / n_TOTAL << inc_mark_hit;
		logout.record() << "[Result]\tMAE = " << current.bm_MAE / n_TOTAL
				<< ", Best-MAE = " << best.bm_MAE / n_TOTAL << inc_mark_mae;
		logout.record() << "[Result]\tRMSE= " << sqrt(current.bm_RMSE / n_TOTAL)
			<< ", Best-RMSE= " << sqrt(best.bm_RMSE / n_TOTAL) << inc_mark_rmse;
	}

	virtual void result()
	{
		int n_TOTAL = dm.ds_test.size();
		logout.record() << "[Task]\tBasic Predict Performance Task";
		logout.record() << "[Result]\tBest-HIT-Ratio = " << best.bm_HIT / n_TOTAL;
		logout.record() << "[Result]\tBest-MAE = " << best.bm_MAE / n_TOTAL;
		logout.record() << "[Result]\tBest-RMSE= " << sqrt(best.bm_RMSE / n_TOTAL);
	}
};