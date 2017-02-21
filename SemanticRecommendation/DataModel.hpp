#pragma once
#include "Import.hpp"
#include "DataSet.hpp"
#include "File.hpp"
#include "Logging.hpp"

class DataModel
{
public:
	const DataSet& ds;

public:
	vector<tuple<int, int, int>> dataset;
	vector<set<int>> ds_test_hit;
	vector<set<int>> ds_train_hit;
	vector<tuple<int, int, int>> ds_train;
	vector<tuple<int, int, int>> ds_test;

public:
	int n_user;
	int n_item;
	int n_rate;

public:
	vec user_filter;
	vec item_filter;
	vec ratings_filter;

public:
	DataModel(const DataSet& ds, function<bool(void)> fn_check_train)
		:ds(ds), n_user(0), n_item(0), n_rate(0)
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

		ds_test_hit.resize(n_user);
		for (auto i = ds_test.begin(); i != ds_test.end(); ++i)
		{
			if (get<2>(*i) > (n_rate - 1) / 2)
				ds_test_hit[get<0>(*i)].insert(get<1>(*i));
		}

		ds_train_hit.resize(n_user);
		for (auto i = ds_train.begin(); i != ds_train.end(); ++i)
		{
			if (get<2>(*i) > (n_rate - 1) / 2)
				ds_train_hit[get<0>(*i)].insert(get<1>(*i));
		}
	}
};