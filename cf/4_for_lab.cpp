#include <iostream>
#include <vector>
#include <cassert>
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

vector<double> add(vector<double> &&x, vector<double> &&y) {
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

vector<double>
restoreAfterNormalization(vector<double> &&w, vector<double> &fMin, vector<double> &fMax,
                          double yMin, double yMax) {
    return restoreAfterNormalization(w, fMin, fMax, yMin, yMax);
}

double checkOnTest(vector<vector<double>> &d, vector<double> &classes, vector<double> &w) {
    int n = d.size(), m = d[0].size();


    vector<double> prediction(n);
    for (int i = 0; i < n; i++) {
        prediction[i] = scalar(w, d[i]);
    }

    return SMAPE(classes, prediction);
}

double checkOnTest(vector<vector<double>> &d, vector<double> &classes, vector<double> &&w) {
    return checkOnTest(d, classes, w);
}

vector<double> addUselessInAnswer(vector<double> &w, set<int> &useless);

vector<double>
doShit(vector<vector<double>> &d, vector<double> &classes,
       vector<double> &featureMin, vector<double> &featureMax,
       double classesMin, double classesMax,
       set<int> &useless,
       double ridge, double lasso,
       vector<vector<double>> &d_test, vector<double> &classes_test) {
    int n = d.size(), m = d[0].size() - 1;

    vector<double> w(m + 1);
    for (double &weight : w) {
        weight = ((double) rand() / RAND_MAX - 0.5) / m;
    }

    cout << "self, test\n";

    int i = 0;
    while (i < 2000) {
        i++;
        double step = 1.0 / i;

        if (i % 10 == 0) {
            step = 0.1;
        }

        int randomObj = rand() % n;

        w = sub(
                sub(
                        mulConst(w, 1.0 - ridge * step),
                        mulConst(gradientStohastic(d, w, classes, randomObj), step)),
                mulConst(sign(w), lasso)
        );

        double smape_self = checkOnTest(d, classes, w);
        double smape_test = checkOnTest(d_test, classes_test,
                                        restoreAfterNormalization(addUselessInAnswer(w, useless), featureMin,
                                                                  featureMax, classesMin, classesMax));
        cout << smape_self << ", " << smape_test << '\n';
    }

    w = addUselessInAnswer(w, useless);
    vector<double> restoredW = restoreAfterNormalization(w, featureMin, featureMax, classesMin, classesMax);

    return restoredW;
}

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
        vector<double> restoredW = doShit(d, classes,
                                          featureMin, featureMax,
                                          classesMin, classesMax,
                                          uselessFeatures,
                                          reg, lasso,
                                          d_test, classes_test);
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

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << setprecision(20);

    string file;
    cin >> file;

    runOnTestFile(file.c_str(), 1e-4, 1e-9, 1);
}