#pragma once
#include "Import.hpp"
#include "DataSet.hpp"
#include "DataModel.hpp"
#include "File.hpp"
#include "Logging.hpp"

class Model
{
public:
	const DataModel& dm;

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
	Model(const DataModel& ds)
		:dm(ds)
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

		logout.record() << "[DataSet]\t" << dm.ds.name;
	}

public:
	virtual double probability(const tuple<int, int, int>& datum) = 0;
	virtual double infer(const int user, const int item) = 0;
	virtual void train_once(const tuple<int, int, int>& datum) = 0;
	virtual void save(FormatFile & file) = 0;
	virtual void load(FormatFile & file) = 0;

public:
	void save(const string& path)
	{
		FormatFile& file = make_fout(path);
		save(file);
		file.close();
	}

	void load(const string& path)
	{
		FormatFile& file = make_fin(path);
		load(file);
		file.close();
	}

	virtual int discrete_infer(const int user, const int item)
	{
		return int(0.5 + infer(user, item));
	}

public:
	double probability_condition(const int user, const int item, const int rating)
	{
		double part = 0;
		for (auto i = 1; i < dm.n_rate; ++i)
		{
			part += probability_simple(user, item, i);
		}

		return probability_simple(user, item, rating) / part;
	}

	double probability_simple(const int user, const int item, const int rating)
	{
		return probability(make_tuple(user, item, rating));
	}

public:
	virtual void train_kernel()
	{
#pragma omp parallel for
		for (auto i = dm.ds_train.begin(); i != dm.ds_train.end(); ++i)
		{
			train_once(*i);
		}
	}

	void train(const int epos, function<void(int)> fn_round_check = [&](int epos){})
	{
		logout.record() << "Start to Train";

		for (auto ind_turn = 0; ind_turn < epos; ++ind_turn)
		{
			logout.record() << "Round @ " << ind_turn;

			train_kernel();

			fn_round_check(ind_turn);
			logout.flush();
		}

		logout.record() << "Train Over";
	}
};