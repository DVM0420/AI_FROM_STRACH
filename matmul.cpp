#include "cpplib.h"
#include "matmul.h"
#include "matadd.h"
#include "activition.h"

using namespace std;

float mat0[] = {1, 2, 3, 4, 5, 6};
float mat1[] = {7, 8, 9, 10, 11, 12};

void print_matrix(const matrix& m) {
    if (m.row == 0 || m.col == 0 || m.mat == nullptr) {
        std::cout << "Ma tran rong hoac khong hop le!\n";
        return;
    }

    std::ios_base::fmtflags old_flags = std::cout.flags();
    std::streamsize old_prec = std::cout.precision();

    std::cout << std::fixed << std::setprecision(2);
    for (uint64_t i = 0; i < m.row; ++i) {
        for (uint64_t j = 0; j < m.col; ++j) {
            std::cout << std::setw(8) << m.mat[i * m.col + j] << " ";
        }
        std::cout << "\n";
    }

    std::cout.flags(old_flags);
    std::cout.precision(old_prec);
}


int main(int argc, char const *argv[])
{
	matrix A;
	A.row = 2;
	A.col = 3;
	vector<float>Av;
	Av.assign(mat0, mat0 + 6);
	A.mat=Av.data();

	matrix B;
	B.row = 2;
	B.col = 3;
	vector<float>Bv;
	Bv.assign(mat1, mat1 + 6);
	B.mat=Bv.data();

	matrix C;
	C.row = 2;
	C.col = 3;
	vector<float>Cv;
	Cv.resize(C.row*C.col);
	C.mat = Cv.data();
	matadd(&A, &B, &C);

	std::cout << "A + B =\n";
	print_matrix(C);


	std::cout<<"OK";
	return 0;
}


