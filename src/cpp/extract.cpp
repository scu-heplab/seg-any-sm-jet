#include "extract.h"

RootExtract::RootExtract(string path)
{
	TString inputfile(path);

	this->chain = new TChain("Delphes");
	this->chain->Add(inputfile);
	this->rootFilePath = path;
	this->treeReader = new ExRootTreeReader(this->chain);
	this->eventNums = this->treeReader->GetEntries();

	this->particleBranch = this->treeReader->UseBranch("Particle");
}

void RootExtract::AssignFinalState(vector<int>& pid, string path)
{
	int particleNum;
	GenParticle* particle;
	map<int, int> initMapIdMapping;
	map<int, GenParticle*> initMap;
	map<int, bool> isSureMap;
	map<pair<int, int>, vector<GenParticle*>> finalMap;

	ofstream out(path);

	for (size_t entry = 0; entry < this->eventNums; entry++)
	{
		this->treeReader->ReadEntry(entry);
		particleNum = this->particleBranch->GetEntries();

		Search(pid, initMap);

		for (auto& init : initMap)
		{
			particle = init.second;
			initMapIdMapping[init.first] = init.first;
			while (particle->D1 == particle->D2 && particle->D1 > 1)
			{
				particle = (GenParticle*)this->particleBranch->At(particle->D1);
				if (particle->PID == init.second->PID)
				{
					initMapIdMapping[init.first] = init.second->D1;
					init.second = particle;
				}
				else break;
			}
			isSureMap[init.first] = true;
			finalMap[pair<int, int>(initMapIdMapping[init.first], init.second->PID)] = { init.second };
		}

		for (int i = 0; i < particleNum; i++)
		{
			particle = (GenParticle*)this->particleBranch->At(i);

			// cout << i << " " << particle->Status << " " << particle->PID << " " << particle->D1 << " " << particle->D2 << " " << particle->M1 << " " << particle->M2 << endl;

			if (particle->Status == 1)
			{
				bool isSure = true;

				finalMap[pair<int, int>(0, 0)].emplace_back(particle);
				for (auto& init : initMap)
				{
					int check = CheckMother(i, initMapIdMapping[init.first]);
					if (check == 1) finalMap[pair<int, int>(initMapIdMapping[init.first], 0)].emplace_back(particle);
					else if (check == -1)
					{
						isSure = false;
						isSureMap[init.first] = false;
					}
				}
				if (!isSure) finalMap[pair<int, int>(0, -1)].emplace_back(particle);
			}
		}

		for (auto& init : initMap) if (!isSureMap[init.first]) finalMap[pair<int, int>(-initMapIdMapping[init.first], init.second->PID)] = { init.second };

		for (auto& fmp : finalMap)
		{
			for (auto& part : fmp.second)
			{
				double x = part->X, y = part->Y, z = part->Z;
				double px = part->Px, py = part->Py, pz = part->Pz;
				double d0 = sqrt(pow(py * (py * x - px * y) + pz * (pz * x - px * z), 2) + pow(px * (px * y - py * x) + pz * (pz * y - py * z), 2)) / (px * px + py * py + pz * pz);

				out << entry << " " << fmp.first.first << " " << fmp.first.second << " ";
				out << part->Px << " " << part->Py << " " << part->Pz << " " << part->E << " " << abs(part->Charge) << " " << d0 << endl;
			}
		}

		initMap.clear();
		finalMap.clear();
		isSureMap.clear();
		initMapIdMapping.clear();

		printf("\r%ld/%ld", entry + 1, this->eventNums);
	}
	printf("\n");
}

void RootExtract::Search(vector<int>& pid, map<int, GenParticle*>& initMap)
{
	int particleNum = this->particleBranch->GetEntries();

	for (int i = 0; i < particleNum; i++)
	{
		GenParticle* mother;
		GenParticle* particle = (GenParticle*)this->particleBranch->At(i);

		for (int id : pid)
		{
			if (particle->PID == id)
			{
				int index = i;
				bool done = false;

				while (!done)
				{
					int m1 = particle->M1, m2 = particle->M2;

					if ((m1 == m2 && m1 > 1) || (m1 > 1 && m2 < 0))
					{
						mother = (GenParticle*)this->particleBranch->At(m1);
						if (mother->PID == id)
						{
							index = m1;
							particle = mother;
						}
						else done = true;
					}
					else if (m1 < m2 && m1 > 1)
					{
						int status = abs(particle->Status);

						if ((81 <= status && status <= 86) || (101 <= status && status <= 106))
						{
							for (int j = m1; j <= m2; j++)
							{
								mother = (GenParticle*)this->particleBranch->At(j);
								if (mother->PID == id)
								{
									index = j;
									particle = mother;
									break;
								}
							}
							done = mother->PID != id;
						}
						else
						{
							mother = (GenParticle*)this->particleBranch->At(m1);
							if (mother->PID == id)
							{
								index = m1;
								particle = mother;
							}
							else
							{
								mother = (GenParticle*)this->particleBranch->At(m2);
								if (mother->PID == id)
								{
									index = m2;
									particle = mother;
								}
								else done = true;
							}
						}
					}
					else if (m1 > m2 && m2 > 1)
					{
						mother = (GenParticle*)this->particleBranch->At(m1);
						if (mother->PID == id)
						{
							index = m1;
							particle = mother;
						}
						else
						{
							mother = (GenParticle*)this->particleBranch->At(m2);
							if (mother->PID == id)
							{
								index = m2;
								particle = mother;
							}
							else done = true;
						}
					}
					else done = true;
				}

				if (particle->PT > 50) initMap[index] = particle;
			}
		}
	}
}

int RootExtract::CheckMother(int particleID, int motherID)
{
	if (particleID == motherID) return 1;
	else
	{
		int m1, m2;
		GenParticle* particle = (GenParticle*)this->particleBranch->At(particleID);

		m1 = particle->M1;
		m2 = particle->M2;
		if (m1 == m2)
		{
			if (m1 < 0) return 0;
			else return CheckMother(m1, motherID);
		}
		else if (m1 >= 0 && m2 < 0) return CheckMother(m1, motherID);
		else if (m1 < m2 && m1 >= 0)
		{
			int status = abs(particle->Status);

			if ((81 <= status && status <= 86) || (101 <= status && status <= 106) || status == 2)
			{
				int flag = 10;

				for (int i = m1; i <= m2; i++)
				{
					int check = CheckMother(i, motherID);

					if (check == -1) return -1;
					else if (flag == 10) flag = check;
					else if (flag != check) return -1;
				}
				return flag;
			}
			else
			{
				int flag1 = CheckMother(m1, motherID);
				int flag2 = CheckMother(m2, motherID);

				if (flag1 == flag2) return flag1;
				else return -1;
			}
		}
		else if (m1 > m2 && m2 >= 0)
		{
			int flag1 = CheckMother(m1, motherID);
			int flag2 = CheckMother(m2, motherID);

			if (flag1 == flag2) return flag1;
			else return -1;
		}
		else return 0;
	}
}

RootExtract::~RootExtract()
{

	delete this->treeReader;
	delete this->chain;

	this->treeReader = nullptr;
	this->chain = nullptr;
	this->particleBranch = nullptr;
}