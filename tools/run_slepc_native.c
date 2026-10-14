#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <slepceps.h>

typedef struct {
  PetscInt n;
  PetscInt nnz;
  PetscInt *rowptr;
  PetscInt *colind;
  PetscScalar *values;
} CsrMatrix;

static PetscErrorCode read_matrix_market(const char *path, CsrMatrix *matrix) {
  FILE *file = fopen(path, "r");
  if (!file) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "cannot open Matrix Market file %s: %s", path, strerror(errno));
  }
  char line[4096];
  if (!fgets(line, sizeof(line), file) || strncmp(line, "%%MatrixMarket matrix coordinate", 32) != 0) {
    fclose(file);
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "unsupported Matrix Market header in %s", path);
  }
  PetscBool symmetric = (strstr(line, "symmetric") != NULL) ? PETSC_TRUE : PETSC_FALSE;
  do {
    if (!fgets(line, sizeof(line), file)) {
      fclose(file);
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "missing Matrix Market dimensions in %s", path);
    }
  } while (line[0] == '%');
  PetscInt rows, cols, entries;
  if (sscanf(line, "%" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT, &rows, &cols, &entries) != 3 || rows != cols) {
    fclose(file);
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "matrix must be square in %s", path);
  }
  PetscInt capacity = symmetric ? 2 * entries : entries;
  PetscInt *coo_rows = NULL, *coo_cols = NULL;
  PetscScalar *coo_values = NULL;
  PetscMalloc3(capacity, &coo_rows, capacity, &coo_cols, capacity, &coo_values);
  PetscInt count = 0;
  for (PetscInt entry = 0; entry < entries; ++entry) {
    if (!fgets(line, sizeof(line), file)) {
      PetscFree3(coo_rows, coo_cols, coo_values);
      fclose(file);
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "truncated Matrix Market data in %s", path);
    }
    PetscInt row, col;
    PetscReal value;
    if (sscanf(line, "%" PetscInt_FMT " %" PetscInt_FMT " %lf", &row, &col, &value) != 3) {
      PetscFree3(coo_rows, coo_cols, coo_values);
      fclose(file);
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "unsupported Matrix Market value in %s", path);
    }
    coo_rows[count] = row - 1;
    coo_cols[count] = col - 1;
    coo_values[count] = value;
    ++count;
    if (symmetric && row != col) {
      coo_rows[count] = col - 1;
      coo_cols[count] = row - 1;
      coo_values[count] = value;
      ++count;
    }
  }
  fclose(file);

  matrix->n = rows;
  matrix->nnz = count;
  PetscCall(PetscCalloc1(rows + 1, &matrix->rowptr));
  PetscCall(PetscMalloc2(count, &matrix->colind, count, &matrix->values));
  for (PetscInt index = 0; index < count; ++index) {
    ++matrix->rowptr[coo_rows[index] + 1];
  }
  for (PetscInt row = 0; row < rows; ++row) {
    matrix->rowptr[row + 1] += matrix->rowptr[row];
  }
  PetscInt *cursor = NULL;
  PetscCall(PetscMalloc1(rows, &cursor));
  PetscCall(PetscArraycpy(cursor, matrix->rowptr, rows));
  for (PetscInt index = 0; index < count; ++index) {
    PetscInt destination = cursor[coo_rows[index]]++;
    matrix->colind[destination] = coo_cols[index];
    matrix->values[destination] = coo_values[index];
  }
  PetscFree(cursor);
  PetscFree3(coo_rows, coo_cols, coo_values);
  return 0;
}

static PetscErrorCode distribute_matrix(const char *path, MPI_Comm comm, Mat *result, PetscInt *global_size, PetscInt *row_start, PetscInt *row_stop) {
  PetscMPIInt rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  CsrMatrix root_matrix = {0};
  if (rank == 0) {
    PetscCall(read_matrix_market(path, &root_matrix));
  }
  PetscInt n = rank == 0 ? root_matrix.n : 0;
  MPI_Bcast(&n, 1, MPIU_INT, 0, comm);
  PetscInt start = (n * rank) / size;
  PetscInt stop = (n * (rank + 1)) / size;
  PetscInt local_rows = stop - start;
  PetscInt local_nnz = 0;
  PetscInt *row_counts = NULL, *row_displacements = NULL;
  PetscInt *nnz_counts = NULL, *nnz_displacements = NULL;
  PetscInt *global_row_lengths = NULL;
  if (rank == 0) {
    PetscCall(PetscMalloc4(size, &row_counts, size, &row_displacements, size, &nnz_counts, size, &nnz_displacements));
    PetscCall(PetscMalloc1(n, &global_row_lengths));
    for (PetscInt row = 0; row < n; ++row) {
      global_row_lengths[row] = root_matrix.rowptr[row + 1] - root_matrix.rowptr[row];
    }
    for (PetscMPIInt worker = 0; worker < size; ++worker) {
      PetscInt worker_start = (n * worker) / size;
      PetscInt worker_stop = (n * (worker + 1)) / size;
      row_counts[worker] = worker_stop - worker_start;
      row_displacements[worker] = worker_start;
      nnz_counts[worker] = root_matrix.rowptr[worker_stop] - root_matrix.rowptr[worker_start];
      nnz_displacements[worker] = root_matrix.rowptr[worker_start];
    }
  }
  MPI_Scatter(nnz_counts, 1, MPIU_INT, &local_nnz, 1, MPIU_INT, 0, comm);
  PetscInt *local_lengths = NULL;
  PetscCall(PetscMalloc1(PetscMax(local_rows, 1), &local_lengths));
  MPI_Scatterv(global_row_lengths, row_counts, row_displacements, MPIU_INT, local_lengths, local_rows, MPIU_INT, 0, comm);
  PetscInt *local_colind = NULL;
  PetscScalar *local_values = NULL;
  PetscCall(PetscMalloc1(PetscMax(local_nnz, 1), &local_colind));
  PetscCall(PetscMalloc1(PetscMax(local_nnz, 1), &local_values));
  MPI_Scatterv(rank == 0 ? root_matrix.colind : NULL, nnz_counts, nnz_displacements, MPIU_INT, local_colind, local_nnz, MPIU_INT, 0, comm);
  MPI_Scatterv(rank == 0 ? root_matrix.values : NULL, nnz_counts, nnz_displacements, MPIU_SCALAR, local_values, local_nnz, MPIU_SCALAR, 0, comm);

  PetscInt *rowptr = NULL;
  PetscCall(PetscMalloc1(local_rows + 1, &rowptr));
  rowptr[0] = 0;
  for (PetscInt row = 0; row < local_rows; ++row) {
    rowptr[row + 1] = rowptr[row] + local_lengths[row];
  }
  PetscFree(local_lengths);
  PetscInt *diag_nnz = NULL;
  PetscInt *offdiag_nnz = NULL;
  PetscCall(PetscCalloc1(PetscMax(local_rows, 1), &diag_nnz));
  PetscCall(PetscCalloc1(PetscMax(local_rows, 1), &offdiag_nnz));
  for (PetscInt row = 0; row < local_rows; ++row) {
    for (PetscInt index = rowptr[row]; index < rowptr[row + 1]; ++index) {
      PetscInt column = local_colind[index];
      if (column >= start && column < stop) ++diag_nnz[row];
      else ++offdiag_nnz[row];
    }
  }
  PetscCall(MatCreateAIJ(comm, local_rows, local_rows, n, n, 0, diag_nnz, 0, offdiag_nnz, result));
  for (PetscInt row = 0; row < local_rows; ++row) {
    PetscInt global_row = start + row;
    PetscCall(MatSetValues(*result, 1, &global_row, rowptr[row + 1] - rowptr[row], local_colind + rowptr[row], local_values + rowptr[row], INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(*result, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*result, MAT_FINAL_ASSEMBLY));
  PetscBool is_symmetric;
  PetscCall(MatIsSymmetric(*result, 1e-12, &is_symmetric));
  if (!is_symmetric) {
    SETERRQ(comm, PETSC_ERR_ARG_WRONG, "native driver accepts only real symmetric matrices");
  }
  PetscCall(MatSetOption(*result, MAT_SYMMETRIC, PETSC_TRUE));
  PetscFree(rowptr);
  PetscFree(diag_nnz);
  PetscFree(offdiag_nnz);
  PetscFree(local_colind);
  PetscFree(local_values);
  if (rank == 0) {
    PetscFree4(row_counts, row_displacements, nnz_counts, nnz_displacements);
    PetscFree(global_row_lengths);
    PetscFree(root_matrix.rowptr);
    PetscFree2(root_matrix.colind, root_matrix.values);
  }
  *global_size = n;
  *row_start = start;
  *row_stop = stop;
  return 0;
}

static PetscErrorCode run_solver(const char *matrix_path, const char *output_path, PetscInt nev, PetscReal sigma, PetscReal tolerance) {
  MPI_Comm comm = PETSC_COMM_WORLD;
  PetscMPIInt rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  PetscLogDouble solve_started, solve_finished, elapsed_seconds;
  PetscTime(&solve_started);
  PetscInt n, row_start, row_stop;
  Mat A;
  PetscCall(distribute_matrix(matrix_path, comm, &A, &n, &row_start, &row_stop));
  EPS eps;
  PetscCall(EPSCreate(comm, &eps));
  PetscCall(EPSSetOperators(eps, A, NULL));
  PetscCall(EPSSetProblemType(eps, EPS_HEP));
  PetscCall(EPSSetDimensions(eps, nev, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(EPSSetTolerances(eps, tolerance, PETSC_DEFAULT));
  PetscCall(EPSSetTarget(eps, sigma));
  PetscCall(EPSSetWhichEigenpairs(eps, size == 1 ? EPS_TARGET_MAGNITUDE : EPS_SMALLEST_REAL));
  ST st;
  PetscCall(EPSGetST(eps, &st));
  PetscCall(STSetType(st, size == 1 ? STSINVERT : STSHIFT));
  PetscCall(STSetShift(st, sigma));
  KSP ksp;
  PC pc;
  PetscCall(STGetKSP(st, &ksp));
  if (size == 1) {
    PetscCall(KSPSetType(ksp, KSPPREONLY));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCLU));
    PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERPETSC));
  } else {
    PetscCall(KSPSetType(ksp, KSPGMRES));
    PetscCall(KSPSetTolerances(ksp, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, 500));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCJACOBI));
  }
  PetscCall(EPSSetFromOptions(eps));
  PetscCall(EPSSolve(eps));
  PetscInt converged;
  PetscCall(EPSGetConverged(eps, &converged));
  if (converged < nev) {
    SETERRQ(comm, PETSC_ERR_NOT_CONVERGED, "SLEPc converged only %" PetscInt_FMT " of %" PetscInt_FMT " requested eigenpairs", converged, nev);
  }
  PetscReal matrix_norm;
  PetscCall(MatNorm(A, NORM_FROBENIUS, &matrix_norm));
  Vec xr, xi, ax;
  PetscCall(MatCreateVecs(A, &xr, &xi));
  PetscCall(VecDuplicate(xr, &ax));
  PetscReal *residuals = NULL;
  PetscScalar *values = NULL;
  PetscCall(PetscMalloc2(nev, &residuals, nev, &values));
  for (PetscInt index = 0; index < nev; ++index) {
    PetscScalar kr, ki;
    PetscCall(EPSGetEigenpair(eps, index, &kr, &ki, xr, xi));
    values[index] = kr;
    PetscCall(MatMult(A, xr, ax));
    PetscCall(VecAXPY(ax, -kr, xr));
    PetscReal numerator, denominator;
    PetscCall(VecNorm(ax, NORM_2, &numerator));
    PetscCall(VecNorm(xr, NORM_2, &denominator));
    residuals[index] = numerator / PetscMax((matrix_norm + PetscAbsScalar(kr)) * denominator, PETSC_SMALL);
  }
  PetscTime(&solve_finished);
  PetscLogDouble local_elapsed = solve_finished - solve_started;
  MPI_Allreduce(&local_elapsed, &elapsed_seconds, 1, MPI_DOUBLE, MPI_MAX, comm);
  MatInfo matrix_info;
  PetscCall(MatGetInfo(A, MAT_GLOBAL_SUM, &matrix_info));
  if (rank == 0) {
    FILE *file = fopen(output_path, "w");
    if (!file) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "cannot open output file %s: %s", output_path, strerror(errno));
    }
    PetscReal maximum = 0.0;
    const char *spectral_transform = size == 1 ? "sinvert" : "shift";
    const char *selection_rule = size == 1 ? "target_magnitude" : "smallest_real";
    for (PetscInt index = 0; index < nev; ++index) maximum = PetscMax(maximum, residuals[index]);
    fprintf(file, "{\n");
    fprintf(file, "  \"backend\": \"slepc-native-mpi\",\n");
    fprintf(file, "  \"distributed_single_matrix\": true,\n");
    fprintf(file, "  \"mpi_size\": %d,\n", (int)size);
    fprintf(file, "  \"matrix_shape\": [%" PetscInt_FMT ", %" PetscInt_FMT "],\n", n, n);
    fprintf(file, "  \"matrix_nnz\": %.0f,\n", (double)matrix_info.nz_used);
    fprintf(file, "  \"requested_k\": %" PetscInt_FMT ",\n", nev);
    fprintf(file, "  \"requested_sigma\": %.17g,\n", (double)sigma);
    fprintf(file, "  \"converged\": %" PetscInt_FMT ",\n", converged);
    fprintf(file, "  \"spectral_transform\": \"%s\",\n", spectral_transform);
    fprintf(file, "  \"selection_rule\": \"%s\",\n", selection_rule);
    fprintf(file, "  \"elapsed_seconds\": %.9f,\n", (double)elapsed_seconds);
    fprintf(file, "  \"max_relative_residual\": %.17g,\n", (double)maximum);
    fprintf(file, "  \"eigenvalues\": [");
    for (PetscInt index = 0; index < nev; ++index) fprintf(file, "%s%.17g", index ? ", " : "", (double)PetscRealPart(values[index]));
    fprintf(file, "],\n  \"residuals\": [");
    for (PetscInt index = 0; index < nev; ++index) fprintf(file, "%s%.17g", index ? ", " : "", (double)residuals[index]);
    fprintf(file, "]\n}\n");
    fclose(file);
    PetscPrintf(PETSC_COMM_SELF, "wrote %s (ranks=%d, max_relative_residual=%.3e)\n", output_path, (int)size, (double)maximum);
  }
  PetscFree2(residuals, values);
  PetscCall(VecDestroy(&ax));
  PetscCall(VecDestroy(&xr));
  PetscCall(VecDestroy(&xi));
  PetscCall(EPSDestroy(&eps));
  PetscCall(MatDestroy(&A));
  return 0;
}

int main(int argc, char **argv) {
  PetscErrorCode ierr = SlepcInitialize(&argc, &argv, NULL, NULL);
  if (ierr) return 1;
  if (argc < 3) {
    PetscPrintf(PETSC_COMM_WORLD, "usage: %s matrix.mtx output.json [k] [sigma] [tol]\n", argv[0]);
    SlepcFinalize();
    return 2;
  }
  PetscInt nev = argc > 3 ? (PetscInt)strtol(argv[3], NULL, 10) : 4;
  PetscReal sigma = argc > 4 ? (PetscReal)strtod(argv[4], NULL) : 0.0;
  PetscReal tolerance = argc > 5 ? (PetscReal)strtod(argv[5], NULL) : 1e-10;
  ierr = run_solver(argv[1], argv[2], nev, sigma, tolerance);
  PetscErrorCode finalize_error = SlepcFinalize();
  return ierr || finalize_error ? 1 : 0;
}
