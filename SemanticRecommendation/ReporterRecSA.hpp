#pragma once
#include "Import.hpp"
#include "DetailedConfig.hpp"
#include "Benchmark.hpp"
#include "SemanticAnalysis.hpp"
#include "Visualization.hpp"
#include "Utils.hpp"
#include <boost/algorithm/string.hpp>

void report_figure_scatter(RecSA* model)
{
	for (auto un = 0; un < model->get_unit_number(); ++un)
	{
		mat img = ones(1000, 1000);
		Draw dw(img);
		for (auto u = 0; u < model->dm.n_user; ++u)
		{
			vec pt = model->infer_user_distribution(u, un);
			dw.box(pt(0) * 999, pt(1) * 999, 3);
		}

		GrayPPM ppm(img);
		FormatFile& file = make_fout(time_namer() + " Visual Rate-4 Unit-" - un + ".ppm");
		file << ppm;
		make_close(file);
	}
}

void report_semantic_user(RecSA* model, int user_property)
{
	vector<vector<int>> words;
	vector<int>	word_cn;
	map<string, int> name_occupi;
	ifstream fin("H:\\Data\\Recommendation\\MovieLens\\movielens-100k\\u.user");
	while (!fin.eof())
	{
		string strline;
		vector<string> strattr;

		getline(fin, strline);
		split(strattr, strline, boost::is_any_of("|"));

		if (strattr.size() < 4)
			continue;

		vector<int> elem(3);
		int age = strattr[1]/age;

		elem[0] = (strattr[3] == "entertainment" ? 0 : (strattr[3] == "artist" ? 1 : 4)) * 2
			+ (strattr[2] == "M" ? 1 : 0);
		elem[1] = (strattr[2] == "M" ? 0 : 1);
		if (name_occupi.find(strattr[3]) == name_occupi.end())
		{
			name_occupi[strattr[3]] = name_occupi.size();
		}
		elem[2] = name_occupi[strattr[3]];

		words.push_back(elem);
	}
	fin.close();

	word_cn.push_back(4);
	word_cn.push_back(2);
	word_cn.push_back(name_occupi.size());

	for (auto un = 0; un < model->get_unit_number(); ++un)
	{
		mat img_r = ones(515, 515);
		mat img_g = ones(515, 515);
		mat img_b = ones(515, 515);
		Draw dw_r(img_r);
		Draw dw_g(img_g);
		Draw dw_b(img_b);

		for (auto u = 1; u < model->dm.n_user; ++u)
		{
			int up = words[u - 1][user_property];
			if (up != 1 && up != 2)
				continue;

			double r = 0;
			double g = 0;
			double b = 0;

			switch (up)
			{
			case 0:r = 1.0; break;
			case 1:g = 1.0; break;
			case 2:b = 1.0; break;
			case 3:r = 1.0; g = 1.0; break;
			}
			vec pt = model->infer_user_distribution(u, un);
			dw_r.mark(pt(0) * 500, pt(1) * 500, 5, r);
			dw_g.mark(pt(0) * 500, pt(1) * 500, 5, g);
			dw_b.mark(pt(0) * 500, pt(1) * 500, 5, b);
		}

		ColorPPM ppm(img_r, img_g, img_b);
		FormatFile& file = make_fout(time_namer() + " Visual User Property-" - user_property -
			" Unit-" - un + ".ppm");
		file << ppm;
		make_close(file);
	}
}

void report_semantic_item(RecSA* model)
{
	vector<vector<int>> words;
	ifstream fin("H:\\Data\\Recommendation\\MovieLens\\movielens-100k\\u.item");
	while (!fin.eof())
	{
		string strline;
		vector<string> strattr;

		getline(fin, strline);
		split(strattr, strline, boost::is_any_of("|"));

		if (strattr.size() < 4)
			continue;

		vector<int> elem(20);
		for (auto i = 4; i < 23; ++i)
		{
			int age;
			age = strattr[i] / age;
			elem[i - 4] = age;
		}
		words.push_back(elem);
	}
	fin.close();

	for (int un = 1; un<5; ++un)
	for (auto d = 0; d < 19; d++)
	{
		mat img_r = ones(515, 515);
		mat img_g = ones(515, 515);
		mat img_b = ones(515, 515);
		Draw dw_r(img_r);
		Draw dw_g(img_g);
		Draw dw_b(img_b);

		for (auto u = 1; u < model->dm.n_item; ++u)
		{
			int up = 0;
			if (words[u - 1][d] == 1)
			{
				up = 1;
			}
			else
			{
				continue;
			}

			double r = (up == 0) ? 1 : 0;
			double g = 0;
			double b = (up == 0) ? 0 : 1;

			vec pt = model->infer_item_distribution(u, un);
			dw_r.mark(pt(0) * 499, pt(1) * 499, 5, r);
			dw_g.mark(pt(0) * 499, pt(1) * 499, 5, g);
			dw_b.mark(pt(0) * 499, pt(1) * 499, 5, b);
		}

		ColorPPM ppm(img_r, img_g, img_b);
		FormatFile& file = make_fout(time_namer() + " Visual Item Unit-" - d + ".ppm");
		file << ppm;
		make_close(file);

	}
}
