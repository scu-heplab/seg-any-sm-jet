#include <string>
#include <vector>
#include <sstream>
#include "extract.h"

void print_usage(const char* prog_name) {
	std::cerr << "Usage: " << prog_name << " [options]\n"
		<< "\n"
		<< "Options:\n"
		<< "  -h, --help           Show this help message\n"
		<< "  -i, --input  <path>  Path to the input ROOT file (required)\n"
		<< "  -o, --output <path>  Path to the output .dat file (required)\n"
		<< "  -p, --pids   <list>  Comma-separated list of particle PIDs (required).\n"
		<< "                       Example: \"25,6,-6,5,-5\"\n"
		<< std::endl;
}

bool parse_pids(const std::string& pids_str, std::vector<int>& pid_list) {
	std::stringstream ss(pids_str);
	std::string item;

	while (std::getline(ss, item, ',')) {
		if (item.empty()) {
			continue;
		}

		std::stringstream item_ss(item);
		int pid;

		item_ss >> pid;

		if (item_ss.fail()) {
			std::cerr << "Error: Invalid character found in PID part '" << item << "'. Please provide only comma-separated integers." << std::endl;
			return false;
		}

		char c;
		if (item_ss >> c) {
			std::cerr << "Error: Trailing characters '" << c << "' found after number in PID part '" << item << "'." << std::endl;
			return false;
		}

		pid_list.push_back(pid);
	}
	return true;
}

int main(int argc, char* argv[]) {
	std::string input_path;
	std::string output_path;
	std::string pids_str;

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if (arg == "-h" || arg == "--help") {
			print_usage(argv[0]);
			return 0;
		}
		if (i + 1 < argc) {
			if (arg == "-i" || arg == "--input") {
				input_path = argv[++i];
			}
			else if (arg == "-o" || arg == "--output") {
				output_path = argv[++i];
			}
			else if (arg == "-p" || arg == "--pids") {
				pids_str = argv[++i];
			}
		}
	}

	if (input_path.empty() || output_path.empty() || pids_str.empty()) {
		std::cerr << "Error: Missing required arguments." << std::endl;
		print_usage(argv[0]);
		return 1;
	}

	std::vector<int> pid_list;
	std::vector<int> base_pids = { 21, 1, 2, 3, 4, -1, -2, -3, -4 };
	pid_list.insert(pid_list.end(), base_pids.begin(), base_pids.end());

	std::vector<int> specific_pids;
	if (!parse_pids(pids_str, specific_pids)) {
		return 1;
	}
	pid_list.insert(pid_list.begin(), specific_pids.begin(), specific_pids.end());

	std::cout << "Input ROOT file: " << input_path << std::endl;
	std::cout << "Final PID list: ";
	for (size_t i = 0; i < pid_list.size(); ++i) {
		std::cout << pid_list[i] << (i == pid_list.size() - 1 ? "" : ", ");
	}
	std::cout << std::endl;
	std::cout << "Saving result to: " << output_path << std::endl;

	try {
		RootExtract extract(input_path);
		extract.AssignFinalState(pid_list, output_path);
		std::cout << "Processing complete." << std::endl;
	}
	catch (const std::exception& e) {
		std::cerr << "An error occurred during processing: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}