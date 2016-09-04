/*
 ******************************************************************
 * HISTORY
 * 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
 *      Prepared for 15-681, Fall 1994.
 *
 ******************************************************************
 */

#ifndef _BACKPROP_H_

#define _BACKPROP_H_

#define BIGRND 0x7fffffff

/*** The neural network data structure.  The network is assumed to
     be a fully-connected feedforward three-layer network.
     Unit 0 in each layer of units is the threshold unit; this means
     that the remaining units are indexed from 1 to n, inclusive.
 ***/

typedef struct {
	int input_n;
	double *input_units;
	double **input_weights;
	double **input_prev_weights;
}Input_Layer;

typedef struct {
	int hidden_n;
	double *hidden_units;
	double **hidden_weights;
	double **hidden_prev_weights;
	double *hidden_delta;
}Hidden_Layer;

typedef struct{
	int output_n;
	double *output_units;
	double *output_delta;
}Output_Layer;

typedef struct {
	int hiddenlayer_n;
	Input_Layer *in_layerpoint;
	Hidden_Layer **hidden_layerlist;
	Output_Layer *out_layerpoint;
	double *target;
}DPNN;



typedef struct {
  int input_n;                  /* number of input units */
  //int hidden_n;                 /* number of hidden units */
  int output_n;                 /* number of output units */

  /*新修改多层隐含节点*/
  //int hidden_levels;           //隐含节点层数
  int hidden_l1n;
  int hidden_l2n;
  int hidden_l3n;
  /*以上新加入*/


  double *input_units;          /* the input units */
  //double *hidden_units;         /* the hidden units */

  /*新加*/
  double *hidden_l1_units;
  double *hidden_l2_units;
  double *hidden_l3_units;

  double *output_units;         /* the output units */

  //double *hidden_delta;         /* storage for hidden unit error */

  /*新加*/
  double *hidden_l1_delta;
  double *hidden_l2_delta;
  double *hidden_l3_delta;

  double *output_delta;         /* storage for output unit error */

  double *target;               /* storage for target vector */

  double **input_weights;       /* weights from input to hidden layer */
  //double **hidden_weights;      /* weights from hidden to output layer */

  /*新加*/
  double **hidden_l1_weights;
  double **hidden_l2_weights;
  double **hidden_l3_weights;

                                /*** The next two are for momentum ***/
  double **input_prev_weights;  /* previous change on input to hidden wgt */
  //double **hidden_prev_weights; /* previous change on hidden to output wgt */

  /*新加*/
  double **hidden_l1_prev_weights;
  double **hidden_l2_prev_weights;
  double **hidden_l3_prev_weights;

} BPNN;


/*** User-level functions ***/

void dpnn_initialize(unsigned int seed);

DPNN *dpnn_create(int n_in, int n_hidden, int n_out, int *hidden_n_array);

void dpnn_free(DPNN *net);


void dpnn_train(DPNN *net,double eta, double dmomentum, double *eo, double *eh);

void dpnn_feedforward(DPNN *net);

void dpnn_save(DPNN *net,char *filename);

DPNN *dpnn_read(char *filename);

#endif
