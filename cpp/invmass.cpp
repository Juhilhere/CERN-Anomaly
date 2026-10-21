// Recompute dimuon invariant mass from pt, eta, phi (same physics as src/a1_dimuon_spectrum.py).
// Build: g++ -O2 -std=c++17 -o invmass.exe invmass.cpp
// Run:   invmass.exe ../data/Dimuon_SingleMu.csv masses_cpp.txt
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr double kMuonMass = 0.1056584;  // GeV

struct FourVector {
    double e, px, py, pz;
};

FourVector fromPtEtaPhi(double pt, double eta, double phi) {
    const double px = pt * std::cos(phi);
    const double py = pt * std::sin(phi);
    const double pz = pt * std::sinh(eta);
    return {std::sqrt(px * px + py * py + pz * pz + kMuonMass * kMuonMass), px, py, pz};
}

double invariantMass(const FourVector& a, const FourVector& b) {
    const double e = a.e + b.e, px = a.px + b.px, py = a.py + b.py, pz = a.pz + b.pz;
    const double m2 = e * e - px * px - py * py - pz * pz;
    return m2 > 0.0 ? std::sqrt(m2) : 0.0;  // rounding can make m2 slightly negative
}

std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) out.push_back(cell);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: invmass <input.csv> <output.txt>\n";
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "cannot open " << argv[1] << "\n";
        return 1;
    }
    std::string header;
    std::getline(in, header);
    const auto names = split(header);
    auto col = [&](const std::string& n) -> size_t {
        for (size_t i = 0; i < names.size(); ++i)
            if (names[i] == n) return i;
        std::cerr << "missing column " << n << "\n";
        std::exit(1);
    };
    const size_t ipt1 = col("pt1"), ieta1 = col("eta1"), iphi1 = col("phi1");
    const size_t ipt2 = col("pt2"), ieta2 = col("eta2"), iphi2 = col("phi2");

    std::ofstream out(argv[2]);
    out << std::setprecision(12);
    std::string line;
    size_t n = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto c = split(line);
        const auto a = fromPtEtaPhi(std::stod(c[ipt1]), std::stod(c[ieta1]), std::stod(c[iphi1]));
        const auto b = fromPtEtaPhi(std::stod(c[ipt2]), std::stod(c[ieta2]), std::stod(c[iphi2]));
        out << invariantMass(a, b) << "\n";
        ++n;
    }
    std::cout << "wrote " << n << " masses\n";
    return 0;
}
