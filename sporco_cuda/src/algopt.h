//  Author: Gustavo Silva <gustavo.silva@pucp.edu.pe>

/*******************************************/
/****        Algorithm Options          ****/
/*******************************************/

typedef struct AlgOpt {

  int MaxMainIter;

  float rho;
  float RhoRsdlRatio;
  float RhoScaling;
  float RhoRsdlTarget;
  float RelaxParam;
  float AbsStopTol;
  float RelStopTol;

  float *L1Weight;
  int L1_WEIGHT_M_SIZE;
  int L1_WEIGHT_ROW_SIZE;
  int L1_WEIGHT_COL_SIZE;

  int *Weight;
  int nWeight;

  int Verbose;
  int NonNegCoef;
  int NoBndryCross;
  int AuxVarObj;
  int HighMemSolve;
  int AutoRho;
  int AutoRhoScaling;
  int AutoRhoPeriod;
  int StdResiduals;

  int IMG_ROW_SIZE;
  int IMG_COL_SIZE;
  int DICT_M_SIZE;
  int DICT_ROW_SIZE;
  int DICT_COL_SIZE;

  float *GrdWeight;
  int WEIGHT_SIZE;

  int device;

} AlgOpt;
