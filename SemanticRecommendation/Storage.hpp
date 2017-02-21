#pragma once
#include "Import.hpp"

class Storage
{
protected:
	map<string, fstream*>	storage_system;

public:
	static Storage* global_system;

public:
	void open_file(const string& file_name, const string& file_path, const ios::ios_base::open_mode open_mode = 3)
	{
		storage_system[file_name] = new fstream(file_path.c_str(), open_mode);
		if (storage_system[file_name]->bad() || storage_system[file_name]->fail())
			throw exception("Bad File Path.");
	}

	void flush_file(const string& file_name)
	{
		auto ifile = storage_system.find(file_name);
		if (ifile == storage_system.end())
			throw exception("No File Found.");

		ifile->second->flush();
	}

	void close_file(const string& file_name)
	{
		auto ifile = storage_system.find(file_name);
		if (ifile == storage_system.end())
			throw exception("No File Found.");

		ifile->second->close();
		storage_system.erase(ifile);
	}

	fstream& find_file(const string& file_name)
	{
		auto ifile = storage_system.find(file_name);
		if (ifile == storage_system.end())
			throw exception("No File Found.");

		return *ifile->second;
	}
};

Storage* Storage::global_system = new Storage;