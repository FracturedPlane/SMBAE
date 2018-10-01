#include "caffe/caffe.hpp"
#include "caffe/solver.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include <memory>

typedef double tData;
const int gXSize = 1;
const int gYSize = 1;
const double gMinX = 0;
const double gMaxX = 1;
const int gNumTrainData = 2048;
const int gNumTestData = 128;
const int gNumPredSamples = 256;
const std::string gDataLayerName = "input";
const std::string gDeployNetFile = "data/test_net.prototxt";
const std::string gSolverFile = "data/test_solver.prototxt";

void LoadNet(const std::string& net_file, std::unique_ptr<caffe::Net<tData>>& out_net)
{
	out_net = std::unique_ptr<caffe::Net<tData>>(new caffe::Net<tData>(gDeployNetFile, caffe::TEST));
}

void LoadSolver(const std::string& solver_file, std::unique_ptr<caffe::Solver<tData>>& out_solver)
{
	caffe::SolverParameter solver_param;
	caffe::ReadProtoFromTextFileOrDie(solver_file, &solver_param);
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	out_solver = std::unique_ptr<caffe::Solver<tData>>(caffe::SolverRegistry<tData>::CreateSolver(solver_param));
}

void BuildData(int n, std::vector<tData>& out_xs, std::vector<tData>& out_ys)
{
	out_xs.resize(n * gXSize);
	out_ys.resize(n * gYSize);
	int idx = 0;

	for (int i = 0; i < n; ++i)
	{
		double total = 0;
		for (int j = 0; j < gXSize; ++j)
		{
			double r = static_cast<double>(rand() % RAND_MAX) / RAND_MAX;
			r = r * (gMaxX - gMinX) + gMinX;
			out_xs[i * gXSize + j] = r;
			total += r;
		}

		for (int j = 0; j < gYSize; ++j)
		{
			double x = total;
			double val = (std::cos(5 * x) - 0.75)*(std::sin(5 * x) + 0.75);
			out_ys[i * gYSize + j] = val;
		}
	}
}

void LoadData(std::vector<tData>& xs, std::vector<tData>& ys, const boost::shared_ptr<caffe::Net<tData>>& out_net)
{
	auto data_layer = boost::static_pointer_cast<caffe::MemoryDataLayer<tData>>(out_net->layer_by_name(gDataLayerName));
	int n = xs.size() / gXSize;
	data_layer->Reset(xs.data(), ys.data(), n);
}

void Train(std::vector<tData>& xs, std::vector<tData>& ys, const std::unique_ptr<caffe::Solver<tData>>& out_solver)
{
	auto train_net = out_solver->net();
	LoadData(xs, ys, train_net);
	out_solver->Solve();
}

void CopyParams(const std::unique_ptr<caffe::Solver<tData>>& solver, std::unique_ptr<caffe::Net<tData>>& out_net)
{
	auto train_net = solver->net();
	caffe::NetParameter params;
	train_net->ToProto(&params);
	out_net->CopyTrainedLayersFrom(params);
}

void EvalNet(const std::unique_ptr<caffe::Net<tData>>& net, const std::vector<tData>& x, std::vector<tData>& out_y)
{
	const std::vector<caffe::Blob<tData>*>& input_blobs = net->input_blobs();
	const auto& input_blob = input_blobs[0];
	tData* input_data = input_blob->mutable_cpu_data();

	for (int i = 0; i < input_blob->count(); ++i)
	{
		input_data[i] = x[i];
	}

	const std::vector<caffe::Blob<tData>*>& result_arr = net->Forward();

	const caffe::Blob<tData>* result = result_arr[result_arr.size() - 1];
	const tData* result_data = result->cpu_data();

	out_y.resize(result->count());
	for (int i = 0; i < result->count(); ++i)
	{
		out_y[i] = result_data[i];
	}
}

void TestNet(std::vector<tData>& xs, std::vector<tData>& ys, const std::unique_ptr<caffe::Net<tData>>& net)
{
	double err = 0;
	std::vector<tData> curr_x(gXSize);
	std::vector<tData> curr_y(gYSize);
	std::vector<tData> pred_y(gYSize);

	int n = xs.size() / gXSize;
	for (int i = 0; i < n; ++i)
	{
		std::memcpy(curr_x.data(), xs.data() + i * gXSize, gXSize * sizeof(tData));
		std::memcpy(curr_y.data(), ys.data() + i * gYSize, gYSize * sizeof(tData));
		EvalNet(net, curr_x, pred_y);

		for (int j = 0; j < gYSize; ++j)
		{
			double diff = curr_y[j] - pred_y[j];
			err += diff * diff;
		}
	}

	err /= n;
	printf("Average Error: %.5f\n", err);
}


void OutputStuff(std::vector<tData>& xs, std::vector<tData>& ys, const std::unique_ptr<caffe::Net<tData>>& net)
{
	const std::string test_data_file = "output/test_data.txt";
	const std::string pred_data_file = "output/pred_data.txt";
	FILE* test_data_f = nullptr;
	FILE* pred_data_f = nullptr;

	test_data_f = std::fopen(test_data_file.c_str(), "w");
	for (size_t i = 0; i < xs.size(); ++i)
	{
		for (int j = 0; j < gXSize; ++j)
		{
			double val = xs[i * gXSize + j];
			fprintf(test_data_f, "%.5f\t", val);
		}

		for (int j = 0; j < gYSize; ++j)
		{
			double val = ys[i * gYSize + j];
			fprintf(test_data_f, "%.5f\t", val);
		}

		fprintf(test_data_f, "\n");
	}
	std::fclose(test_data_f);
	test_data_f = nullptr;


	pred_data_f = std::fopen(pred_data_file.c_str(), "w");
	std::vector<double> curr_x(gXSize);
	std::vector<double> curr_y;

	for (size_t i = 0; i < gNumPredSamples; ++i)
	{
		for (int j = 0; j < gXSize; ++j)
		{
			double val = static_cast<double>(i) / (gNumPredSamples - 1) * (gMaxX - gMinX) + gMinX;
			curr_x[j] = val;
			fprintf(pred_data_f, "%.5f\t", val);
		}

		EvalNet(net, curr_x, curr_y);
		for (int j = 0; j < gYSize; ++j)
		{
			double val = curr_y[j];
			fprintf(pred_data_f, "%.5f\t", val);
		}

		fprintf(pred_data_f, "\n");
	}
	std::fclose(pred_data_f);
	pred_data_f = nullptr;
}

void main()
{
	std::unique_ptr<caffe::Solver<tData>> solver;
	LoadSolver(gSolverFile, solver);

	std::vector<tData> train_xs;
	std::vector<tData> train_ys;
	BuildData(gNumTrainData, train_xs, train_ys);

	Train(train_xs, train_ys, solver);

	std::unique_ptr<caffe::Net<tData>> net;
	LoadNet(gDeployNetFile, net);
	CopyParams(solver, net);

	std::vector<tData> test_xs;
	std::vector<tData> test_ys;
	BuildData(gNumTestData, test_xs, test_ys);
	TestNet(test_xs, test_ys, net);

	OutputStuff(test_xs, test_ys, net);
}