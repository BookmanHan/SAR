#pragma once
#include "Import.hpp"
#include "DataModel.hpp"
#include "Model.hpp"
#include "Task.hpp"

typedef tuple<int, int, int> recomendation_datum_t;

class RecommendationDataModel
	:public DataModel<recomendation_datum_t>
{
public:
	int n_user;
	int n_item;
	int n_rate;

public:
	vec user_filter;
	vec item_filter;
	vec ratings_filter;

public:
	RecommendationDataModel(const DataSet& ds, function<bool(void)> fn_check_train, const string& sdescprion)
		:DataModel<recomendation_datum_t>(ds, fn_check_train, sdescprion), n_user(0), n_item(0), n_rate(0)
	{
		ifstream fin(ds.url, ios::binary);
		while (!fin.eof())
		{
			int u, i, r;
			fin.read((char*)&u, sizeof(int));
			fin.read((char*)&i, sizeof(int));
			fin.read((char*)&r, sizeof(int));

			n_user = max(n_user, u);
			n_item = max(n_item, i);
			n_rate = max(n_rate, r);

			dataset.push_back(make_tuple(u, i, r));
			if (fn_check_train())
			{
				ds_train.push_back(make_tuple(u, i, r));
			}
			else
			{
				ds_test.push_back(make_tuple(u, i, r));
			}
		}
		fin.close();

		++n_user;
		++n_item;
		++n_rate;

		cout << "DataSet Loaded" << endl;

		logout.record() << "[Data Description] Training Set Size = " << ds_train.size();
		logout.record() << "[Data Description] Testing Set Size = " << ds_test.size();

		user_filter = ones(n_user) * mini_factor;
		item_filter = ones(n_item) * mini_factor;
		ratings_filter = ones(n_rate) * mini_factor;
	}
};

class RecommendationModel
	:public Model<recomendation_datum_t>
{
public:
	const RecommendationDataModel& dm;

public:
	vec avg_user;
	vec avg_item;
	vec avg_rate;
	vec cnt_user;
	vec cnt_item;
	vec cnt_rate;
	vec prior_rate;
	vector<int> user_low_bound;
	vector<int> item_low_bound;

public:
	RecommendationModel(RecommendationDataModel& dm)
		:Model<recomendation_datum_t>((DataModel<recomendation_datum_t>*)&dm), dm(dm)
	{
		avg_user = zeros(dm.n_user);
		avg_item = zeros(dm.n_item);
		avg_rate = zeros(dm.n_rate);
		cnt_user = zeros(dm.n_user);
		cnt_item = zeros(dm.n_item);
		cnt_rate = zeros(dm.n_rate);

		user_low_bound.resize(dm.n_user);
		std::fill(user_low_bound.begin(), user_low_bound.end(), dm.n_rate);
		item_low_bound.resize(dm.n_item);
		std::fill(item_low_bound.begin(), item_low_bound.end(), dm.n_rate);

		for (auto i = dm.ds_train.begin(); i != dm.ds_train.end(); ++i)
		{
			avg_user[get<0>(*i)] += get<2>(*i);
			avg_item[get<1>(*i)] += get<2>(*i);
			avg_rate[get<2>(*i)] += get<2>(*i);
			++cnt_user[get<0>(*i)];
			++cnt_item[get<1>(*i)];
			++cnt_rate[get<2>(*i)];

			user_low_bound[get<0>(*i)] = min(get<2>(*i), user_low_bound[get<0>(*i)]);
			item_low_bound[get<1>(*i)] = min(get<2>(*i), item_low_bound[get<1>(*i)]);
		}

		avg_user = avg_user / cnt_user;
		avg_item = avg_item / cnt_item;
		avg_rate = avg_rate / cnt_rate;
		prior_rate = cnt_rate / accu(cnt_rate);
	}
};

class BasicPredictPerformanceTask
	:public Task<recomendation_datum_t>
{
protected:
	RecommendationDataModel& dm;

public:
	struct Performance
	{
		double bm_HIT;
		double bm_MAE;
		double bm_RMSE;
	} best, current;
	const bool open_stein_effect;

public:
	BasicPredictPerformanceTask(DataModel<recomendation_datum_t>& dm, bool open_stein_effect = false)
		:Task<recomendation_datum_t>(&dm), open_stein_effect(open_stein_effect)
	{
		best.bm_HIT = 0;
		best.bm_MAE = 1e10;
		best.bm_RMSE = 1e10;
	}

	BasicPredictPerformanceTask(Model<recomendation_datum_t> *model, bool open_stein_effect = false)
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
			++n_TOTAL;
			double pred_rating = model->infer(get<0>(*i), get<1>(*i)) - mean_bias;

			if (int(pred_rating + 0.5) == get<2>(*i))
				++current.bm_HIT;

			current.bm_MAE += abs(model->discrete_infer(get<0>(*i), get<1>(*i)) - get<2>(*i));
			current.bm_RMSE += (pred_rating - get<2>(*i)) * (pred_rating - get<2>(*i));
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

	virtual void result(Logging& loging = logout)
	{
		int n_TOTAL = dm.ds_test.size();
		loging.record() << "[Task]\tBasic Predict Performance Task";
		loging.record() << "[Result]\tBest-HIT-Ratio = " << best.bm_HIT / n_TOTAL;
		loging.record() << "[Result]\tBest-MAE = " << best.bm_MAE / n_TOTAL;
		loging.record() << "[Result]\tBest-RMSE= " << sqrt(best.bm_RMSE / n_TOTAL);
	}
};