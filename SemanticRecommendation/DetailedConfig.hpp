#pragma once
#include "Import.hpp"
#include "DataSet.hpp"

const string report_path = "H:\\Report\\Semantics\\";
const string model_path = "H:\\Model\\Semantics\\";
const string image_reader = "\"C:\\Program Files (x86)\\2345Soft\\2345Pic\\2345PicEditor.exe\" ";
const int parallel_number = 16;

const DataSet ML100K("ML100K", "H:\\Data\\Recommendation\\ML100K.binary");
const DataSet ML1M("ML1M", "H:\\Data\\Recommendation\\ML1M.binary");
const DataSet ML10M("ML10M", "H:\\Data\\Recommendation\\ML10M.binary");
const DataSet ML20M("ML20M", "H:\\Data\\Recommendation\\ML20M.binary");

const function<bool(void)> SEPRATION_10_PEC = [](){if (rand() % 100 < 10) return true; else return false; };
const function<bool(void)> SEPRATION_30_PEC = [](){if (rand() % 100 < 30) return true; else return false; };
const function<bool(void)> SEPRATION_90_PEC = [](){if (rand() % 100 < 90) return true; else return false; };
const function<bool(void)> SEPRATION_50_PEC = [](){if (rand() % 100 < 50) return true; else return false; };
const function<bool(void)> SEPRATION_70_PEC = [](){if (rand() % 100 < 70) return true; else return false; };
const function<bool(void)> SEPRATION_CV5 = [](){if (rand() % 100 < 80) return true; else return false; };
const function<bool(void)> SEPRATION_CV10 = [](){if (rand() % 100 < 90) return true; else return false; };