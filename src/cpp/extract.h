#pragma once

#include <fstream>
#include <iostream>

#include <TMath.h>
#include <TClonesArray.h>

#include <classes/DelphesClasses.h>
#include <ExRootAnalysis/ExRootTreeReader.h>

using namespace std;

class RootExtract
{
public:
	RootExtract(string path);
	void AssignFinalState(vector<int>& pid, string path);
	~RootExtract();
private:
	size_t eventNums;
	TChain* chain;
	string rootFilePath;
	ExRootTreeReader* treeReader;
	TClonesArray* particleBranch;

	void Search(vector<int>& pid, map<int, GenParticle*>& initMap);
	int CheckMother(int particleID, int motherID);
};