#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <iomanip>

using namespace std;

vector<bool> toVec(int i, int sz) {
    vector<bool> res(sz);
    int index = 0;
    while (i > 0) {
        res[index] = i % 2;
        i /= 2;
        index++;
    }

    return res;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << setprecision(20);

    int m;
    cin >> m;

    int pow2m = pow(2, m);

    vector<int> rows(pow2m);

    int onesCount = 0;
    for (int i = 0; i < pow2m; i++) {
        cin >> rows[i];

        if (rows[i] == 1) {
            onesCount++;
        }
    }

    bool inverse = onesCount > 512;

    vector<vector<double>> neurons;
    for (int i = 0; i < pow2m; i++) {
        if ((!inverse && rows[i] == 1) || (inverse && rows[i] == 0)) {
            vector<bool> asVec = toVec(i, m);

            assert(asVec.size() == m);

            vector<double> newNeuron(m + 1);

            onesCount = 0;
            for (int j = 0; j < m; j++) {
                onesCount += asVec[j];
            }

            double oneMul, sub;

            if (onesCount == 0) {
                oneMul = 0;
                sub = -0.5;
            } else {
                oneMul = 1.0 / onesCount;
                sub = oneMul * (onesCount - 1.0) + oneMul / 2.0;
            }

            for (int j = 0; j < m; j++) {
                newNeuron[j] = asVec[j] ? oneMul : -1.0;
            }
            newNeuron[m] = -sub;

            neurons.push_back(newNeuron);
        }
    }

    vector<double> orNeuron(neurons.size() + 1, inverse ? -1 : 1);
    orNeuron[neurons.size()] = inverse ? 0.5 : -0.5;

    neurons.push_back(orNeuron);

    cout << 2 << '\n';
    cout << neurons.size() - 1 << " " << 1 << '\n';
    for (auto& neuron : neurons) {
        for (double val : neuron) {
            cout << val << " ";
        }
        cout << '\n';
    }
}