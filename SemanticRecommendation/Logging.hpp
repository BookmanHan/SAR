#pragma once
#include "Import.hpp"
#include "Utils.hpp"

class Logging
{
protected:
	ofstream fout;

public:
	Logging(const string& base_dir = report_path)
	{
		fout.open((time_namer() + ".txt").c_str());

		const time_t log_time = time(nullptr);
		struct tm* current_time = localtime(&log_time);

		stringstream ss;
		ss << 1900 + current_time->tm_year << "-";
		ss << setfill('0') << setw(2) << current_time->tm_mon + 1 << "-";
		ss << setfill('0') << setw(2) << current_time->tm_mday << " ";
		ss << setfill('0') << setw(2) << current_time->tm_hour << ".";
		ss << setfill('0') << setw(2) << current_time->tm_min << ".";
		ss << setfill('0') << setw(2) << current_time->tm_sec;
		fout << '[' << ss.str() << ']' << '\t' <<"Ready.";;
		cout << '[' << ss.str() << ']' << '\t' << "Ready.";;
	}

	Logging& record()
	{
		const time_t log_time = time(nullptr);
		struct tm* current_time = localtime(&log_time);
		stringstream ss;
		ss << 1900 + current_time->tm_year << "-";
		ss << setfill('0') << setw(2) << current_time->tm_mon + 1 << "-";
		ss << setfill('0') << setw(2) << current_time->tm_mday << " ";
		ss << setfill('0') << setw(2) << current_time->tm_hour << ".";
		ss << setfill('0') << setw(2) << current_time->tm_min << ".";
		ss << setfill('0') << setw(2) << current_time->tm_sec;

		fout << endl;
		fout << '[' << ss.str() << ']' << '\t';
		cout << endl;
		cout << '[' << ss.str() << ']' << '\t';

		return *this;
	}

	Logging& direct_record()
	{
		const time_t log_time = time(nullptr);
		struct tm* current_time = localtime(&log_time);
		stringstream ss;
		ss << 1900 + current_time->tm_year << "-";
		ss << setfill('0') << setw(2) << current_time->tm_mon + 1 << "-";
		ss << setfill('0') << setw(2) << current_time->tm_mday << " ";
		ss << setfill('0') << setw(2) << current_time->tm_hour << ".";
		ss << setfill('0') << setw(2) << current_time->tm_min << ".";
		ss << setfill('0') << setw(2) << current_time->tm_sec;

		fout << endl;
		fout << '[' << ss.str() << ']' << '\t';

		return *this;
	}

	void flush()
	{
		fout.flush();
		cout.flush();
	}

	void redirect(const string& str_suffix = "")
	{
		fout.close();
		fout.open((time_namer() + str_suffix + ".txt").c_str());

		const time_t log_time = time(nullptr);
		struct tm* current_time = localtime(&log_time);

		stringstream ss;
		ss << 1900 + current_time->tm_year << "-";
		ss << setfill('0') << setw(2) << current_time->tm_mon + 1 << "-";
		ss << setfill('0') << setw(2) << current_time->tm_mday << " ";
		ss << setfill('0') << setw(2) << current_time->tm_hour << ".";
		ss << setfill('0') << setw(2) << current_time->tm_min << ".";
		ss << setfill('0') << setw(2) << current_time->tm_sec;
		fout << '[' << ss.str() << ']' << '\t' << "Ready.";;
		cout << '[' << ss.str() << ']' << '\t' << "Ready.";;
	}

public:
	template<typename T>
	Logging& operator << (T things)
	{
		cout << things;
		fout << things;
		return *this;
	}
};


Logging logout;
