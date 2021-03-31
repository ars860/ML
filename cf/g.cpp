#include <iostream>
#include <vector>
#include <set>
#include <cassert>
#include <map>
#include <iomanip>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <algorithm>

using namespace std;

class node {
public:
    int num;

    node *left = nullptr;
    node *right = nullptr;

    int feature;
    double featureVal;

    int target;

    bool isLeaf() {
        return left == nullptr && right == nullptr;
    }

    node(int num, node *left, node *right, int feature, double featureVal) : num(num), left(left), right(right),
                                                                             feature(feature), featureVal(featureVal) {}

    node(int num) : num(num) {}
};

int treeSize(node *n);

pair<
        pair<vector<vector<double> *>, vector<int>>,
        pair<vector<vector<double> *>, vector<int>>>
splitByFeature(int feature, double featureVal, vector<vector<double> *> &d, vector<int> &target) {
    pair<
            pair<vector<vector<double> *>, vector<int>>,
            pair<vector<vector<double> *>, vector<int>>> result;
    vector<vector<double> *> &left = result.first.first;
    vector<int> &leftTarget = result.first.second;

    vector<vector<double> *> &right = result.second.first;
    vector<int> &rightTarget = result.second.second;

    left.reserve(d.size());
    right.reserve(d.size());
    leftTarget.reserve(d.size());
    rightTarget.reserve(d.size());

    for (int i = 0; i < d.size(); i++) {
        if ((*d[i])[feature] <= featureVal) {
            left.push_back(d[i]);
            leftTarget.push_back(target[i]);
        } else {
            right.push_back(d[i]);
            rightTarget.push_back(target[i]);
        }
    }

    return result;
}

pair<vector<vector<double> *>, vector<int>>
sortByFeature(int feature, vector<vector<double> *> &d, vector<int> &target) {
    assert(d.size() == target.size());

    vector<pair<double, pair<vector<double> *, int>>> featureVal2featuresTarget(d.size());
    for (int i = 0; i < d.size(); i++) {
        featureVal2featuresTarget[i] = {(*d[i])[feature],
                                        {d[i], target[i]}};
    }

    sort(featureVal2featuresTarget.begin(), featureVal2featuresTarget.end(),
         [](pair<double, pair<vector<double> *, int>> first, pair<double, pair<vector<double> *, int>> second) {
             return first.first < second.first;
         });

    vector<vector<double> *> res(featureVal2featuresTarget.size());
    vector<int> resTarget(featureVal2featuresTarget.size());

    for (int i = 0; i < featureVal2featuresTarget.size(); i++) {
        res[i] = featureVal2featuresTarget[i].second.first;
        resTarget[i] = featureVal2featuresTarget[i].second.second;
    }

    return {res, resTarget};
}

double GiniIndex(vector<vector<double> *> &part, vector<int> &target) {
    assert(!part.empty());
    assert(part.size() == target.size());

    unordered_map<int, double> target2amount;
    for (int t : target) {
        target2amount[t]++;
    }
    for (auto &t : target2amount) {
        t.second /= target.size();
    }

    double sum = 0.0;
    for (auto &i : target2amount) {
        sum += target2amount[i.first] * target2amount[i.first];
    }

    return 1.0 - sum;
}

double GiniIndexFromTo(int l, int r, vector<int> &targets) {
    unordered_map<int, double> target2amount;
    for (int i = l; i < r; i++) {
        target2amount[targets[i]]++;
    }

    double sum = 0.0;
    for (auto &i : target2amount) {
        sum += target2amount[i.first] * target2amount[i.first] ;
    }

    return 1.0 - sum / (r - l) / (r - l);
}

//double calcFs(int feature, double featureVal,
//              double fs, unordered_set<int> &target_set,
//              vector<vector<double> *> &d, vector<int> &target) {
//    auto leftRight = splitByFeature(feature, featureVal, d, target);
//
//    auto &left = leftRight.first.first;
//    auto &leftTarget = leftRight.first.second;
//    auto &right = leftRight.second.first;
//    auto &rightTarget = leftRight.second.second;
//
//    if (left.empty() || right.empty()) {
//        return -1;
//    }
//
//    double Fs = fs -
//                (double) left.size() / d.size() * GiniIndex(left, leftTarget, target_set) -
//                (double) right.size() / d.size() * GiniIndex(right, rightTarget, target_set);
//
//    return Fs;
//}

node *growTree(vector<vector<double> *> &d, vector<int> &target,
               int maxH,
               vector<pair<double, double>> &feature2minMax,
               int num = 0) {
//    if (maxH == 0) {
//        return nullptr;
//    }

    if (maxH == 0) {
        unordered_map<int, int> target2count;
        for (int t : target) {
            target2count[t]++;
        }

        pair<int, int> maxCount = {-1, -1};
        for (auto &p : target2count) {
            maxCount = max(maxCount, {p.second, p.first});
        }

        node *newNode = new node(num);
        newNode->target = maxCount.second;

        return newNode;
    }

    unordered_set<int> target_set(target.begin(), target.end());
    if (target_set.size() == 1) {
        node *newNode = new node(num);
        newNode->target = *target_set.begin();

        return newNode;
    }

//    vector<unordered_set<double>> feature2vals(d[0]->size());
//    for (vector<double> *x: d) {
//        for (int i = 0; i < x->size(); i++) {
//            feature2vals[i].insert((*x)[i]);
//        }
//    }

    double fs = GiniIndex(d, target);

    pair<double, pair<int, double>> optimalSplit = {-1, {-1, -1}};

    for (int feature = 0; feature < d[0]->size(); feature++) {
//        double l = feature2minMax[feature].first, r = feature2minMax[feature].second;


        auto d_target_sorted = sortByFeature(feature, d, target);
        auto &d_sorted = d_target_sorted.first;
        auto &target_sorted = d_target_sorted.second;

        double featureValuePrev = -1;//(*d_sorted[0])[feature];

        unordered_map<int, double> target2amountLeft;
        unordered_map<int, double> target2amountRight;

        for (int t : target_sorted) {
            target2amountRight[t]++;
        }

        double GiniRight = 0.0;
        for (auto &i : target2amountRight) {
            GiniRight += target2amountRight[i.first] * target2amountRight[i.first];
        }
        double GiniLeft = 0;


        for (int i = 0; i < d_sorted.size(); i++) {
            double featureVal = (*d_sorted[i])[feature];
            if (featureVal == featureValuePrev) {
//                prevsCount++;

                GiniRight = GiniRight - 2 * target2amountRight[target_sorted[i]] + 1;
                GiniLeft = GiniLeft + 2 * target2amountLeft[target_sorted[i]] + 1;
                target2amountLeft[target_sorted[i]]++;
                target2amountRight[target_sorted[i]]--;
                continue;
            }
//            prevsCount = 1;

            int left_size = i, right_size = d_sorted.size() - i;

//            assert(!(left_size == 0 || right_size == 0));
            if (left_size == 0 || right_size == 0) {
                featureValuePrev = featureVal;

                GiniRight = GiniRight - 2 * target2amountRight[target_sorted[i]] + 1;
                GiniLeft = GiniLeft + 2 * target2amountLeft[target_sorted[i]] + 1;
                target2amountLeft[target_sorted[i]]++;
                target2amountRight[target_sorted[i]]--;
                continue;
            }

//            target2amountLeft[target_sorted[i - 1]] += prevsCount;
//            target2amountRight[target_sorted[i - 1]] -= prevsCount;

//            GiniRight = GiniRight - 2 * prevsCount * target2amountRight[target_sorted[i - 1]] + prevsCount * prevsCount;
//            GiniLeft = GiniLeft + 2 * prevsCount * target2amountLeft[target_sorted[i - 1]] + prevsCount * prevsCount;

            double GiniLeftCalculated = 1 - GiniLeft / i / i;
//            double GiniLeftCalculatedTest = GiniIndexFromTo(0, i, target_sorted);
            double GiniRightCalculated = 1 - GiniRight / (target_sorted.size() - i) / (target_sorted.size() - i);
//            double GiniRightCalculatedTest = GiniIndexFromTo(i, target_sorted.size(), target_sorted);

//            assert(GiniLeftCalculated == GiniLeftCalculatedTest);
//            assert(GiniRightCalculated == GiniRightCalculatedTest);

            double Fs = fs -
                        (double) left_size / d.size() * GiniLeftCalculated -
                        (double) right_size / d.size() * GiniRightCalculated;
            optimalSplit = max(optimalSplit, {Fs, {feature, featureValuePrev}});

            featureValuePrev = featureVal;

            GiniRight = GiniRight - 2 * target2amountRight[target_sorted[i]] + 1;
            GiniLeft = GiniLeft + 2 * target2amountLeft[target_sorted[i]] + 1;
            target2amountLeft[target_sorted[i]]++;
            target2amountRight[target_sorted[i]]--;
        }
    }

    auto leftRight = splitByFeature(optimalSplit.second.first, optimalSplit.second.second, d, target);
    auto &left = leftRight.first.first;
    auto &leftTarget = leftRight.first.second;
    auto &right = leftRight.second.first;
    auto &rightTarget = leftRight.second.second;

    node *newNode = new node(num);
    newNode->feature = optimalSplit.second.first;
    newNode->featureVal = optimalSplit.second.second;
    newNode->left = growTree(left, leftTarget, maxH - 1, feature2minMax, num + 1);
    newNode->right = growTree(right, rightTarget, maxH - 1, feature2minMax, num + 1 + treeSize(newNode->left));

    return newNode;
}

int treeSize(node *n) {
    if (n == nullptr) {
        return 0;
    }

    if (n->isLeaf()) {
        return 1;
    }

    return treeSize(n->left) + treeSize(n->right) + 1;
}

void printNodeRec(node *n) {
    if (n == nullptr) {
        return;
    }

//    cout << "num=" << n->num << " ";

    if (n->isLeaf()) {
        cout << "C " << n->target << '\n';
    } else {
        cout << "Q " << n->feature + 1 << " " << n->featureVal << " " << n->left->num << " " << n->right->num << '\n';
        printNodeRec(n->left);
        printNodeRec(n->right);
    }
}

void printTree(node *root) {
    cout << treeSize(root) << '\n';
    printNodeRec(root);
}

int classifier(node *tree, vector<double> &x) {
    while (true) {
        if (tree->isLeaf()) {
            return tree->target;
        }

        if (x[tree->feature] <= tree->featureVal) {
            tree = tree->left;
        } else {
            tree = tree->right;
        }
    }
}

double calcAccuracy(node *tree, vector<vector<double>> &d, vector<int> &target) {
    double accuracy = 0;
    for (int i = 0; i < d.size(); i++) {
        int prediction = classifier(tree, d[i]);

//        cout << "predicted: " << prediction << " actual: " << target[i] << '\n';
        if (prediction == target[i]) {
            accuracy++;
        }
    }
    accuracy /= d.size();
    cout << "Accuracy = " << accuracy;
    return accuracy;
}

bool debug = false;

int main() {
//    auto start_time = std::chrono::high_resolution_clock::now();

    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << setprecision(20);

    if (debug) {
        freopen("g.txt", "r", stdin);
    }

    int m, k, h;
    cin >> m >> k >> h;

    int n;
    cin >> n;

    vector<vector<double>> d(n, vector<double>(m));
    vector<int> target(n);

    for (int i = 0; i < n; i++) {
        int tmp;
        for (int j = 0; j < m; j++) {
            cin >> tmp;
            d[i][j] = tmp;
        }
        cin >> target[i];
    }

    vector<vector<double> *> dptr(n);
    for (int i = 0; i < n; i++) {
        dptr[i] = &d[i];
    }

    vector<pair<double, double>> feature2minMax(d[0].size());
    for (int f = 0; f < d[0].size(); f++) {
        feature2minMax[f] = {d[0][f], d[0][f]};
    }

    for (int i = 0; i < d.size(); i++) {
        for (int f = 0; f < d[0].size(); f++) {
            feature2minMax[f].first = min(feature2minMax[f].first, d[i][f]);
            feature2minMax[f].second = max(feature2minMax[f].second, d[i][f]);
        }
    }

    node *root = growTree(dptr, target, h, feature2minMax, 1);
    printTree(root);

//    auto end_time = std::chrono::high_resolution_clock::now();
//    auto time = end_time - start_time;

    if (debug) {
        calcAccuracy(root, d, target);
        cout << '\n';
    }

//    cout << "took " << time / std::chrono::milliseconds(1) << "ms to run.\n";
}