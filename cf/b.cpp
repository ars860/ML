#include <iostream>
#include <vector>
#include <iomanip>

using namespace std;

int main() {
    int k;

    cin >> k;

    vector<double> p(k);
    vector<double> c(k);
    vector<double> t(k);
    double all = 0;

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            double cmij;
            cin >> cmij;

            all += cmij;
            p[j] += cmij;
            c[i] += cmij;
            if (i == j) {
                t[i] = cmij;
            }
        }
    }

    double weightedF1 = 0;

    for (int i = 0; i < k; i++) {
        double prec = t[i] / p[i],
                recoll = t[i] / c[i],
                f1 = 2 * prec * recoll / (prec + recoll);
        if (p[i] != 0 && c[i] != 0 &&  prec + recoll != 0)
            weightedF1 += f1 * c[i];
    }

    weightedF1 /= all;


    double precw = 0;
    double recallw = 0;

    for (int i = 0; i < k; i++) {
        if (p[i] != 0) {
            precw += t[i] * c[i] / p[i];
            recallw += t[i];
        }
    }
    precw /= all;
    recallw /= all;

    cout << setprecision(9) << 2 * precw * recallw / (precw + recallw) << '\n' << weightedF1;
}
