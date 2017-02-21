#pragma once
#include "Import.hpp"
#include "DetailedConfig.hpp"

template<typename T>
class Identifier
{
public:
	vector<T>	id_to_name;
	map<T, int>	name_to_id;

public:
	const T & operator [] (const int id)
	{
		return id_to_name[id];
	}

	const int operator [] (const T name)
	{
		return name_to_id[id];
	}
};

typedef Identifier<string> IdentifiString;

inline string time_namer()
{
	const time_t log_time = time(nullptr);
	struct tm* current_time = localtime(&log_time);
	
	stringstream ss;
	ss << report_path;
	ss << 1900 + current_time->tm_year << ".";
	ss << setfill('0') << setw(2) << current_time->tm_mon + 1 << ".";
	ss << setfill('0') << setw(2) << current_time->tm_mday << "_";
	ss << setfill('0') << setw(2) << current_time->tm_hour << ".";
	ss << setfill('0') << setw(2) << current_time->tm_min << ".";
	ss << setfill('0') << setw(2) << current_time->tm_sec;

	return ss.str();
}

inline
void init_vector_vec(vector<vec>& item, int capibility, int dim)
{
	item.resize(capibility);
	for_each(item.begin(), item.end(), [&](vec& elem){elem = randu(dim) + mini_factor; elem /= accu(elem); });
}

inline
void init_vector_mat(vector<mat>& item, int capibility, int dim)
{
	item.resize(capibility);
	for_each(item.begin(), item.end(), [&](mat& elem){elem = randu(dim, dim) + mini_factor; elem /= accu(elem); });
}

inline
void zeros_vector_vec(vector<vec>& item, int capibility, int dim)
{
	item.resize(capibility);
	for_each(item.begin(), item.end(), [&](vec& elem){elem = zeros(dim); });
}

inline
void zeros_vector_mat(vector<mat>& item, int capibility, int dim)
{
	item.resize(capibility);
	for_each(item.begin(), item.end(), [&](mat& elem){elem = zeros(dim, dim); });
}

template < typename T>
inline void gradient_normal(T& vec, T& grad)
{
#pragma omp critical
	{
		vec += grad;
		vec = vec % sign(vec);
		vec /= accu(vec);
	}
}