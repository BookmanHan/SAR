#pragma once
#include "Import.hpp"
#include "Model.hpp"
#include "Task.hpp"

class Benchmark
{
protected:
	Model * model;
	Task * task;
	const DataModel& dm;

public:
	Benchmark(DataModel& dm, Model* model, Task* task, int epos)
		:dm(dm), model(model), task(task)
	{
		task->set_model(model);
		task->test();
		model->train(epos - 50, [&](int turn){if (turn % 10 == 0) task->test(); });
		model->train(50, [&](int turn){task->test(); });
	}

	Benchmark(Model* model, Task* task, int epos)
		:dm(model->dm), model(model), task(task)
	{
		task->set_model(model);
		task->test();
		model->train(epos - 50, [&](int turn){if (turn % 10 == 0) task->test(); });
		model->train(50, [&](int turn){task->test(); });
	}

	~Benchmark()
	{
		delete model;
		delete task;
	}

public:
	void result()
	{
		task->result();
	}
};

class TestBed
{
protected:
	vector<tuple<Benchmark*, string, string, string>> bed;
	int epos;
public:
	TestBed(int epos)
		:epos(epos)
	{
		;
	}

	~TestBed()
	{
		for (auto b = bed.begin(); b != bed.end(); ++b)
		{
			delete get<0>(*b);
		}
	}

public:
	void add_benchmark(Model* model, Task* task, const string& str_focus, const string& str_std, const string& str_bk = "")
	{
		bed.push_back(make_tuple(new Benchmark(model, task, epos), str_focus, str_std, str_bk));
	}

	void result(const string& method_descriptions = "")
	{
		logout.redirect(" TestBed Report");
		logout.record() << method_descriptions;
		logout.record() << "[Setting]\tEpos=" << epos;
		logout.record();
		for (auto b = bed.begin(); b != bed.end(); ++b)
		{
			logout.record() << "[Focus]\t"<<get<1>(*b);
			get<0>(*b)->result();
			logout.record() << "[Standard]\t" << get<2>(*b);
			logout.record() << "[Backup]\t" << get<3>(*b);
			logout.record();
		}

		logout.redirect();
	}
};