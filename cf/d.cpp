#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <set>
#include <map>

using namespace std;

void printVec(vector<double> &x) {
    for (double val : x) {
        cout << val << " ";
    }
    cout << '\n';
}

double scalar(vector<double> &x, vector<double> &y) {
    assert(x.size() == y.size());
    double res = 0;
    for (int i = 0; i < x.size(); i++) {
        res += x[i] * y[i];
    }

    return res;
}


vector<double> sub(vector<double> &&x, vector<double> &&y) {
    assert(x.size() == y.size());
    vector<double> res(x.size());
    for (int i = 0; i < x.size(); i++) {
        res[i] = x[i] - y[i];
    }

    return res;
}

vector<double> sub(vector<double> &x, vector<double> &y) {
    assert(x.size() == y.size());
    vector<double> res(x.size());
    for (int i = 0; i < x.size(); i++) {
        res[i] = x[i] - y[i];
    }

    return res;
}

vector<double> sub(vector<double> &&x, vector<double> &y) {
    assert(x.size() == y.size());
    vector<double> res(x.size());
    for (int i = 0; i < x.size(); i++) {
        res[i] = x[i] - y[i];
    }

    return res;
}

vector<double> add(vector<double> &&x, vector<double> &&y) {
    assert(x.size() == y.size());
    vector<double> res(x.size());
    for (int i = 0; i < x.size(); i++) {
        res[i] = x[i] + y[i];
    }

    return res;
}

vector<double> add(vector<double> &x, vector<double> &&y) {
    assert(x.size() == y.size());
    vector<double> res(x.size());
    for (int i = 0; i < x.size(); i++) {
        res[i] = x[i] + y[i];
    }

    return res;
}

vector<double> mulConst(vector<double> &x, double y) {
    vector<double> res(x.size());
    for (int i = 0; i < res.size(); i++) {
        res[i] = x[i] * y;
    }

    return res;
}

vector<double> mulConst(vector<double> &&x, double y) {
    return mulConst(x, y);
}

vector<double> sign(vector<double> &x) {
    vector<double> res(x.size());
    for (int i = 0; i < x.size(); i++) {
        res[i] = x[i] > 0 ? 1 : -1;
    }

    return res;
}

double norma1(vector<double> &x) {
    double res = 0;
    for (double i : x) {
        res += abs(i);
    }

    return res;
}

////
//////TODO: SMAPE?
//////vector<double> gradient(vector<vector<double>> &d, vector<double> &w, vector<double> &y) {
//////    vector<double> sums(d.size(), 0.0);
//////
//////    for (int i = 0; i < d.size(); i++) {
//////        sums[i] = 2.0 * (scalar(d[i], w) - y[i]);
//////    }
//////
//////    vector<double> grad(w.size(), 0.0);
//////
////////    for (int j = 0; j < w.size(); j++) {
////////        for (int i = 0; i < d.size(); i++) {
////////            grad[j] += sums[i] * d[j][i];
////////        }
////////    }
//////
//////    for (int j = 0; j < w.size(); j++) {
//////        for (int i = 0; i < d.size(); i++) {
//////            grad[j] += sums[i] * d[i][j];
//////        }
//////    }
//////
//////    return grad;
//////}
//
//vector<double> gradientSMAPE(vector<vector<double>> &d, vector<double> &w, vector<double> &y) {
//    int n = d.size(), m = w.size();
//    vector<double> sums(d.size(), 0.0);
//
//    for (int i = 0; i < d.size(); i++) {
//        sums[i] = scalar(d[i], w);
//    }
//
//    vector<double> res(m, 0);
//    for (int j = 0; j < m; j++) {
//        for (int i = 0; i < n; i++) {
//            res[j] += (sums[i] - y[i]) * d[i][j] / ((sums[i] - y[i]) * (abs(sums[i]) + abs(y[i]))) -
//                      abs(sums[i] - y[i]) * (sums[i] * d[i][j] / abs(sums[i])) /
//                      ((abs(sums[i]) + abs(y[i])) * (abs(sums[i]) + abs(y[i])));
//        }
//        res[j] /= n;
//    }
//
//    return res;
//}

double SMAPE(vector<double> &prediction, vector<double> &actual) {
    assert(prediction.size() == actual.size());
    double res = 0;

    for (int i = 0; i < prediction.size(); i++) {
        res += abs(prediction[i] - actual[i]) / (abs(prediction[i]) + abs(actual[i]) + 0.00000001);
    }
    res /= prediction.size();

    return res;
}

vector<double> gradientStohastic(vector<vector<double>> &d, vector<double> &w, vector<double> &y, int i) {
    int n = d.size(), m = w.size();

    vector<double> res(m, 0.0);


    double sumsI = scalar(d[i], w);


    for (int j = 0; j < m; j++) {
        res[j] = (sumsI - y[i]) * d[i][j] / (abs(sumsI - y[i]) * (abs(sumsI) + abs(y[i]))) -
                 abs(sumsI - y[i]) * (sumsI * d[i][j] / abs(sumsI)) /
                 ((abs(sumsI) + abs(y[i])) * (abs(sumsI) + abs(y[i])));
        res[j] /= n;
    }

    return res;
}

vector<double>
restoreAfterNormalization(vector<double> &w, vector<double> &fMin, vector<double> &fMax,
                          double yMin, double yMax) {
    int fMinSize = fMin.size(), wSize = w.size();

    assert(w.size() - 1 == fMin.size());
    assert(w.size() - 1 == fMax.size());

    vector<double> res(w.size());
    double free = yMin + (yMax - yMin) * w[w.size() - 1];
    double mul = yMax - yMin;

    for (int i = 0; i < w.size() - 1/*minus fake*/; i++) {
        if (fMax[i] - fMin[i] != 0) {
            res[i] = mul * w[i] / (fMax[i] - fMin[i]);
            free -= w[i] * mul / (fMax[i] - fMin[i]) * fMin[i];
        } else {
            res[i] = mul / fMax[i] * w[i];
        }
    }
    res[w.size() - 1] = free;

    return res;
}

double checkOnTest(vector<vector<double>> &d, vector<double> &classes, vector<double> &w) {
    int n = d.size(), m = d[0].size();


    vector<double> prediction(n);
    for (int i = 0; i < n; i++) {
        prediction[i] = scalar(w, d[i]);
    }

    return SMAPE(classes, prediction);
}

vector<double> addUselessInAnswer(vector<double> &w, set<int> &useless);

int batch_size = 10;

vector<double>
doShit(vector<vector<double>> &d, vector<double> &classes,
       vector<double> &featureMin, vector<double> &featureMax,
       double classesMin, double classesMax,
       set<int> &useless,
       double ridge, double lasso) {
    int n = d.size(), m = d[0].size() - 1;

    vector<double> w(m + 1);
    for (double &weight : w) {
        weight = ((double) rand() / RAND_MAX - 0.5) / m;
    }

    int i = 0;
    while (i < 2000) {
        i++;
        double step = 1.0 / i;

        if (i % 10 == 0) {
            step = 0.1;
        }

//        cout << "Before:\n";
//        double prediction = scalar(w, d[randomObj]);
//        cout << "predicted: (" << prediction << ") actual: (" << classes[randomObj] << ") SMAPE: ("
//             << abs(prediction - classes[randomObj]) / (abs(prediction) + abs(classes[randomObj])) << ")\n";

//        int batch_size = (int)1e7 / 2000 / w.size();

//        vector<double> toSub(w.size(), 0);
//        for (int j = 0; j < batch_size; j++) {
        int randomObj = rand() % n;

//            toSub = add(toSub, mulConst(gradientStohastic(d, w, classes, randomObj), step));

        w = sub(
                sub(
                        mulConst(w, 1.0 - ridge * step),
                        mulConst(gradientStohastic(d, w, classes, randomObj), step)),
                mulConst(sign(w), lasso)
            );

//        }

//        w = sub(
//                sub(
//                        mulConst(w, 1.0 - ridge * step),
//                        toSub),
//                mulConst(sign(w), lasso)
//        );

//        w = sub(w, toSub);

//        cout << "After:\n";
//        prediction = scalar(w, d[randomObj]);
//        cout << "predicted: (" << prediction << ") actual: (" << classes[randomObj] << ") SMAPE: ("
//             << abs(prediction - classes[randomObj]) / (abs(prediction) + abs(classes[randomObj])) << ")\n";
    }

    w = addUselessInAnswer(w, useless);

//    int test = w.size();

//    vector<double> predictions(n);
//    for (int i = 0; i < n; i++) {
//        predictions[i] = scalar(d[i], w);
//    }
//    cout << "Learning SMAPE: (" << SMAPE(predictions, classes) << ")\n";
//    cout << "W before restoration:\n";
//    printVec(w);

    vector<double> restoredW = restoreAfterNormalization(w, featureMin, featureMax, classesMin, classesMax);

//    cout << "W after restoration:\n";
//    printVec(restoredW);

//    printVec(restoredW);

    return restoredW;
}

//pair<double, vector<double>>
//checkOnTestSrednee(vector<vector<double>> &d, vector<double> &classes, vector<double> &featureMin,
//                   vector<double> &featureMax,
//                   double classesMin,
//                   double classesMax, double ridge, double lasso, int attempts, vector<vector<double>> &d_test,
//                   vector<double> &classes_test) {
//    pair<double, vector<double>> optimal = {2, {}};
//    for (int i = 0; i < attempts; i++) {
//        vector<double> w = doShit(d, classes, featureMin, featureMax, classesMin, classesMax, ridge, lasso);
//        double smape = checkOnTest(d_test, classes_test, w);
//        if (smape < optimal.first) {
//            optimal = {smape, w};
//        }
//    }
//
//    return optimal;
//}

vector<double> runOnTestFile(const char *file, double reg, double lasso, int attempts) {
    freopen(file, "r", stdin);
    int n, m;

    cin >> m >> n;

    vector<vector<double>> d(n, vector<double>(m + 1, 1.0));
    vector<double> classes(n);

    vector<double> featureMin(m);
    vector<double> featureMax(m);

    for (int i = 0; i < n; i++) {
        int tmp;
        for (int j = 0; j < m; j++) {
            cin >> tmp;
            d[i][j] = tmp;
        }
        cin >> tmp;
        classes[i] = tmp;
    }

    for (int i = 0; i < m; i++) {
        featureMin[i] = d[0][i];
        featureMax[i] = d[0][i];
    }

    double classesMin = classes[0], classesMax = classes[0];

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            featureMin[j] = min(featureMin[j], d[i][j]);
            featureMax[j] = max(featureMax[j], d[i][j]);
        }
        classesMin = min(classesMin, classes[i]);
        classesMax = max(classesMax, classes[i]);
    }

    set<int> uselessFeatures;
    for (int i = 0; i < m; i++) {
        if (featureMin[i] == featureMax[i]) {
            uselessFeatures.insert(i);
        }
    }

    // normalization
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (featureMax[j] - featureMin[j] == 0) {
//                d[i][j] = 1;
            } else {
                d[i][j] = (d[i][j] - featureMin[j]) / (featureMax[j] - featureMin[j]);
            }
        }
        classes[i] = (classes[i] - classesMin) / (classesMax - classesMin);
    }

    vector<vector<double>> d_without_useless(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m + 1; j++) {
            if (uselessFeatures.find(j) == uselessFeatures.end()) {
                d_without_useless[i].push_back(d[i][j]);
            }
        }
    }
    d = d_without_useless;
//    m = d_without_useless.size();
//    int m_without_useless = d[0].size();

    int n_test;
    cin >> n_test;
    vector<vector<double>> d_test(n_test, vector<double>(m + 1, 1));
    vector<double> classes_test(n_test);

    for (int i = 0; i < n_test; i++) {
        for (int j = 0; j < m; j++) {
            cin >> d_test[i][j];
        }
        cin >> classes_test[i];
    }

    vector<double> mapes;

    srand(time(nullptr));

    for (int i = 1; i <= attempts; i++) {
        vector<double> restoredW = doShit(d, classes, featureMin, featureMax, classesMin, classesMax, uselessFeatures,
                                          reg, lasso);
        double smape = checkOnTest(d_test, classes_test, restoredW);

        mapes.push_back(smape);
    }

    return mapes;
}

vector<double> addUselessInAnswer(vector<double> &w, set<int> &useless) {
    vector<double> withUseless;
    int useless_amount = 0;
    for (int i = 0; i < w.size() + useless.size(); i++) {
        if (useless.find(i) == useless.end()) {
            withUseless.push_back(w[i - useless_amount]);
        } else {
            useless_amount++;
            withUseless.push_back(0);
        }
    }

    return withUseless;
}

bool debug = false;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << setprecision(20);

    if (debug) {
//        vector<double> smapes = runOnTestFile("6.txt", 1e-7, 1e-8, 5);
//        double srednee = 0;
//        for (double d : smapes) {
////            cout << d << " ";
//            srednee += d;
//        }
//        cout << srednee / smapes.size();
//        return 0;

        vector<double> possibilities;
        possibilities.push_back(0);

        for (int i = 2; i < 10; i++) {
            possibilities.push_back(1.0 / pow(10, i));
        }

        vector<const char *> files = {"1.txt", "2.txt", "3.txt", "4.txt", "5.txt", "6.txt", "7.txt"};

        map<string, tuple<double, double, vector<double>>> test2smape;

        map<string, tuple<double, double, double>> fileBest;
        map<string, tuple<double, double, double>> fileSredneeBest;

        for (string file : files) {
            fileBest[file] = {-1, -1, 2};
            fileSredneeBest[file] = {-1, -1, 2};
        }

        int i = 0;
//        double reg = 1e-9, lasso = 0;
        for (double reg : possibilities) {
            for (double lasso : possibilities) {
                for (const char *file : files) {
                    cout << i++;

                    vector<double> smapes = runOnTestFile(file, reg, lasso, 5);

                    double srednee = 0, best = 2;
                    cout << "Test: (" << file << ")\n";
                    for (auto p : smapes) {
                        srednee += p;
                        best = min(p, best);
                    }
                    srednee /= smapes.size();
                    cout << "Reg: (" << reg << ") Lasso: (" << lasso << ") srednee SMAPE: (" << srednee << ")"
                         << ") best SMAPE: (" << best << ")\n";

                    if (!isnan(srednee) && !isinf(srednee)) {
                        auto curBest = fileBest.at(file);
                        auto curSredneeBest = fileSredneeBest.at(file);

                        if (get<2>(curBest) > best) {
                            fileBest[file] = {reg, lasso, best};
                        }
                        if (get<2>(curSredneeBest) > srednee) {
                            fileSredneeBest[file] = {reg, lasso, srednee};
                        }
                    }
                }
            }
        }

        freopen("output", "w", stdout);

        for (auto file2reglassobest : fileBest) {
            cout << "File (" << file2reglassobest.first << ") best best: Reg (" << get<0>(file2reglassobest.second)
                 << ") Lasso (" << get<1>(file2reglassobest.second)
                 << ") SMAPE: (" << get<2>(file2reglassobest.second) << ")\n";
        }

        for (auto file2reglassobest : fileSredneeBest) {
            cout << "File (" << file2reglassobest.first << ") srednee best: Reg (" << get<0>(file2reglassobest.second)
                 << ") Lasso (" << get<1>(file2reglassobest.second)
                 << ") SMAPE: (" << get<2>(file2reglassobest.second) << ")\n";
        }

        return 0;
    }

    int n, m;

//    freopen("4.txt", "r", stdin);
//    cin >> m >> n;
    cin >> n >> m;

    // fake feature for free member
    vector<vector<double>> d(n, vector<double>(m + 1, 1.0));
    vector<double> classes(n);

    vector<double> featureMin(m);
    vector<double> featureMax(m);

    for (int i = 0; i < n; i++) {
        int tmp;
        for (int j = 0; j < m; j++) {
            cin >> tmp;
            d[i][j] = tmp;
        }
        cin >> tmp;
        classes[i] = tmp;
    }

    for (int i = 0; i < m; i++) {
        featureMin[i] = d[0][i];
        featureMax[i] = d[0][i];
    }

    double classesMin = classes[0], classesMax = classes[0];

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            featureMin[j] = min(featureMin[j], d[i][j]);
            featureMax[j] = max(featureMax[j], d[i][j]);
        }
        classesMin = min(classesMin, classes[i]);
        classesMax = max(classesMax, classes[i]);
    }

    set<int> uselessFeatures;
    for (int i = 0; i < m; i++) {
        if (featureMin[i] == featureMax[i]) {
            uselessFeatures.insert(i);
        }
    }

    // normalization
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (featureMax[j] - featureMin[j] == 0) {
//                d[i][j] = 1;
            } else {
                d[i][j] = (d[i][j] - featureMin[j]) / (featureMax[j] - featureMin[j]);
            }
        }
        classes[i] = (classes[i] - classesMin) / (classesMax - classesMin);
    }

    vector<vector<double>> d_without_useless(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m + 1; j++) {
            if (uselessFeatures.find(j) == uselessFeatures.end()) {
                d_without_useless[i].push_back(d[i][j]);
            }
        }
    }
    d = d_without_useless;
//    int m_without_useless = d[0].size();



    srand(time(nullptr));
//    uselessFeatures.clear();
    vector<double> restoredW = doShit(d, classes, featureMin, featureMax, classesMin, classesMax, uselessFeatures,
                                      1e-4,
                                      1e-9);

    printVec(restoredW);
}