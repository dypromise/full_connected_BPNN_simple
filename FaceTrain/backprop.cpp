/*
 ******************************************************************
 * HISTORY
 * 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
 *	Prepared for 15-681, Fall 1994.
 *
 ******************************************************************
 */
#include "stdafx.h"

#include <stdio.h>
#include "backprop.h"
#include <math.h>

#define ABS(x)          (((x) > 0.0) ? (x) : (-(x)))

#define fastcopy(to,from,len)\
{\
  register char *_to,*_from;\
  register int _i,_l;\
  _to = (char *)(to);\
  _from = (char *)(from);\
  _l = (len);\
  for (_i = 0; _i < _l; _i++) *_to++ = *_from++;\
}

/*** Return random number between 0.0 and 1.0 ***/
double drnd()
{
  return ((double) rand() / RAND_MAX);
}

/*** Return random number between -1.0 and 1.0 ***/
double dpn1()
{
  return ((drnd() * 2.0) - 1.0);
}

/*** The squashing function.  Currently, it's a sigmoid. ***/

double squash(double x)
{
  return (1.0 / (1.0 + exp(-x)));
}


/*** Allocate 1d array of doubles ***/

double *alloc_1d_dbl(int n)
{
  double *newptr;

  newptr = (double *) malloc ((unsigned) (n * sizeof (double)));
  if (newptr == NULL) {
    printf("ALLOC_1D_DBL: Couldn't allocate array of doubles\n");
    return (NULL);
  }
  return (newptr);
}


/*** Allocate 2d array of doubles ***/

double **alloc_2d_dbl(int m, int n)
{
  int i;
  double **newptr;

  newptr = (double **) malloc ((unsigned) (m * sizeof (double *)));
  if (newptr == NULL) {
    printf("ALLOC_2D_DBL: Couldn't allocate array of dbl ptrs\n");
    return (NULL);
  }

  for (i = 0; i < m; i++) {
    newptr[i] = alloc_1d_dbl(n);
  }

  return (newptr);
}


/*void bpnn_randomize_weights(double **w,int m, int n)
{
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      w[i][j] = 0.1 * dpn1();
    }
  }
}*/
void dpnn_randomize_weights(double **w, int m, int n)
{
	int i, j;

	for (i = 0; i <= m; i++) {
		for (j = 0; j <= n; j++) {
			w[i][j] = 0.1 * dpn1();
		}
	}
}



/*void bpnn_zero_weights(double **w,int m, int n)
{
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      w[i][j] = 0.0;
    }
  }
}*/
void dpnn_zero_weights(double **w, int m, int n)
{
	int i, j;

	for (i = 0; i <= m; i++) {
		for (j = 0; j <= n; j++) {
			w[i][j] = 0.0;
		}
	}
}



void dpnn_initialize(unsigned int seed)
{
	printf("Random number generator seed: %d\n", seed);
	srand(seed);
}


/*BPNN *bpnn_internal_create_multihiddenlevel(int n_in, int n_hidden_l1, int n_hidden_l2, int n_hidden_l3, int n_out)
{
	BPNN *newnet;

	newnet = (BPNN *)malloc(sizeof(BPNN));
	if (newnet == NULL) {
		printf("BPNN_CREATE: Couldn't allocate neural network\n");
		return (NULL);
	}

	newnet->input_n = n_in;

	newnet->hidden_l1n = n_hidden_l1;
	newnet->hidden_l2n = n_hidden_l2;
	newnet->hidden_l3n = n_hidden_l3;

	newnet->output_n = n_out;

	newnet->input_units = alloc_1d_dbl(n_in + 1);
	newnet->hidden_l1_units = alloc_1d_dbl(n_hidden_l1 + 1);
	newnet->hidden_l2_units = alloc_1d_dbl(n_hidden_l2 + 1);
	newnet->hidden_l3_units = alloc_1d_dbl(n_hidden_l3 + 1);
	newnet->output_units = alloc_1d_dbl(n_out + 1);

	newnet->hidden_l1_delta = alloc_1d_dbl(n_hidden_l1 + 1);
	newnet->hidden_l2_delta = alloc_1d_dbl(n_hidden_l2 + 1);
	newnet->hidden_l3_delta = alloc_1d_dbl(n_hidden_l3 + 1);
	newnet->output_delta = alloc_1d_dbl(n_out + 1);

	newnet->target = alloc_1d_dbl(n_out + 1);

	newnet->input_weights = alloc_2d_dbl(n_in + 1, n_hidden_l1 + 1);
	newnet->hidden_l1_weights = alloc_2d_dbl(n_hidden_l1 + 1, n_hidden_l2 + 1);
	newnet->hidden_l2_weights = alloc_2d_dbl(n_hidden_l2 + 1, n_hidden_l3 + 1);
	newnet->hidden_l3_weights = alloc_2d_dbl(n_hidden_l3 + 1, n_out + 1);


	newnet->input_prev_weights = alloc_2d_dbl(n_in + 1, n_hidden_l1 + 1);
	newnet->hidden_l1_prev_weights = alloc_2d_dbl(n_hidden_l1 + 1, n_hidden_l2 + 1);
	newnet->hidden_l2_prev_weights = alloc_2d_dbl(n_hidden_l2 + 1, n_hidden_l3 + 1);
	newnet->hidden_l3_prev_weights = alloc_2d_dbl(n_hidden_l3 + 1, n_out + 1);

	return (newnet);
}*/
DPNN *dpnn_internal_create_net(int n_in, int n_hiddenlayers, int n_out, int *hidden_layer_n_array)
{
	DPNN *newnet;

	newnet = (DPNN *)malloc(sizeof(DPNN));
	if (newnet == NULL) {
		printf("DPNN_CREATE: Couldn't allocate neural network\n");
		return (NULL);
	}
	printf("allocate new empoty net.");
	newnet->in_layerpoint = (Input_Layer *)malloc(sizeof(Input_Layer));
	newnet->out_layerpoint = (Output_Layer *)malloc(sizeof(Output_Layer));
	newnet->hidden_layerlist = (Hidden_Layer **)malloc(unsigned(n_hiddenlayers * sizeof(Hidden_Layer *)));
	for (int i = 0; i < n_hiddenlayers; i++) {
		newnet->hidden_layerlist[i] = (Hidden_Layer *)malloc(sizeof(Hidden_Layer));
	}
	newnet->target = alloc_1d_dbl(n_out + 1);
	
	newnet->in_layerpoint->input_n = n_in;
	newnet->in_layerpoint->input_units = alloc_1d_dbl(n_in + 1);
	newnet->in_layerpoint->input_weights = alloc_2d_dbl(n_in + 1 , hidden_layer_n_array[0]+1);
	newnet->in_layerpoint->input_prev_weights = alloc_2d_dbl(n_in + 1, hidden_layer_n_array[0] + 1);


	newnet->out_layerpoint->output_n = n_out;
	newnet->out_layerpoint->output_units = alloc_1d_dbl(n_out + 1);
	newnet->out_layerpoint->output_delta = alloc_1d_dbl(n_out + 1);


	for (int i = 0; i < n_hiddenlayers; i++) {
		newnet->hidden_layerlist[i]->hidden_n = hidden_layer_n_array[i];
		newnet->hidden_layerlist[i]->hidden_units= alloc_1d_dbl(hidden_layer_n_array[i] + 1);
		newnet->hidden_layerlist[i]->hidden_delta = alloc_1d_dbl(hidden_layer_n_array[i] + 1);
		if (i != n_hiddenlayers-1) {
			newnet->hidden_layerlist[i]->hidden_weights = alloc_2d_dbl(hidden_layer_n_array[i] + 1, hidden_layer_n_array[i + 1] + 1);
			newnet->hidden_layerlist[i]->hidden_prev_weights = alloc_2d_dbl(hidden_layer_n_array[i] + 1, hidden_layer_n_array[i + 1] + 1);
		}
		else {
			newnet->hidden_layerlist[i]->hidden_weights = alloc_2d_dbl(hidden_layer_n_array[i] + 1, n_out + 1);
			newnet->hidden_layerlist[i]->hidden_prev_weights = alloc_2d_dbl(hidden_layer_n_array[i] + 1, n_out + 1);
		}
	}
	newnet->hiddenlayer_n = n_hiddenlayers;
	return (newnet);
}


void dpnn_free(DPNN *net)
{
	int n1, n2, nh;

	n1 = net->in_layerpoint->input_n;
	n2 = net->out_layerpoint->output_n;
	nh = net->hiddenlayer_n;

	free((char *)net->in_layerpoint->input_units);
	free((char *)net->out_layerpoint->output_units);
	free((char *)net->out_layerpoint->output_delta);

	free((char *)net->target);

	for (int i = 0; i <= n1; i++) {
		free((char *)net->in_layerpoint->input_weights[i]);
		free((char *)net->in_layerpoint->input_prev_weights[i]);
	}
	free((char *)net->in_layerpoint->input_weights);
	free((char *)net->in_layerpoint->input_prev_weights);

	for (int i = 0; i < nh; i++) {
		int n_h = net->hidden_layerlist[i]->hidden_n;
		free((char *)net->hidden_layerlist[i]->hidden_delta);
		free((char *)net->hidden_layerlist[i]->hidden_units);
		for (int j = 0; j < n_h; j++) {
			free((char *)net->hidden_layerlist[i]->hidden_prev_weights[i]);
			free((char *)net->hidden_layerlist[i]->hidden_weights[i]);
		}
		free((char *)net->hidden_layerlist[i]->hidden_prev_weights);
		free((char *)net->hidden_layerlist[i]->hidden_weights);
		free((char *)net->hidden_layerlist[i]);
	}
	free((char *)net->in_layerpoint);
	free((char *)net->out_layerpoint);
	free((char *)net->hidden_layerlist);

	free((char *)net);
}



/*void bpnn_free_mltihiddenlevel(BPNN *net)
{
	int n1, n2_l1, n2_l2, n2_l3, i;

	n1 = net->input_n;
	n2_l1 = net->hidden_l1n;
	n2_l2 = net->hidden_l2n;
	n2_l3 = net->hidden_l3n;

	free((char *)net->input_units);
	free((char *)net->hidden_l1_units);
	free((char *)net->hidden_l2_units);
	free((char *)net->hidden_l3_units);
	free((char *)net->output_units);

	free((char *)net->hidden_l1_delta);
	free((char *)net->hidden_l2_delta);
	free((char *)net->hidden_l3_delta);
	free((char *)net->output_delta);
	free((char *)net->target);

	for (i = 0; i <= n1; i++) {
		free((char *)net->input_weights[i]);
		free((char *)net->input_prev_weights[i]);
	}
	free((char *)net->input_weights);
	free((char *)net->input_prev_weights);

	for (i = 0; i <= n2_l1; i++) {
		free((char *)net->hidden_l1_weights[i]);
		free((char *)net->hidden_l1_prev_weights[i]);
	}
	for (i = 0; i <= n2_l2; i++) {
		free((char *)net->hidden_l2_weights[i]);
		free((char *)net->hidden_l2_prev_weights[i]);
	}
	for (i = 0; i <= n2_l3; i++) {
		free((char *)net->hidden_l3_weights[i]);
		free((char *)net->hidden_l3_prev_weights[i]);
	}
	free((char *)net->hidden_l1_weights);
	free((char *)net->hidden_l2_weights);
	free((char *)net->hidden_l3_weights);
	free((char *)net->hidden_l1_prev_weights);
	free((char *)net->hidden_l2_prev_weights);
	free((char *)net->hidden_l3_prev_weights);

	free((char *)net);
}*/


/*** Creates a new fully-connected network from scratch,
     with the given numbers of input, hidden, and output units.
     Threshold units are automatically included.  All weights are
     randomly initialized.

     Space is also allocated for temporary storage (momentum weights,
     error computations, etc).
***/

/*BPNN *bpnn_create(int n_in, int n_hidden, int n_out)
{

  BPNN *newnet;

  newnet = bpnn_internal_create(n_in, n_hidden, n_out);

//#define INITZERO

#ifdef INITZERO
  bpnn_zero_weights(newnet->input_weights, n_in, n_hidden);
#else
  bpnn_randomize_weights(newnet->input_weights, n_in, n_hidden);
#endif
  bpnn_randomize_weights(newnet->hidden_weights, n_hidden, n_out);
  bpnn_zero_weights(newnet->input_prev_weights, n_in, n_hidden);
  bpnn_zero_weights(newnet->hidden_prev_weights, n_hidden, n_out);

  return (newnet);
}*/

DPNN *dpnn_create(int n_in, int n_hiddenlayers, int n_out, int *a)
{

	DPNN *newnet;

	newnet = dpnn_internal_create_net(n_in, n_hiddenlayers, n_out, a);

	//#define INITZERO

//#ifdef INITZERO
	//dpnn_zero_weights(newnet->in_layerpoint->input_weights, n_in, a[0]);
//#else
	dpnn_randomize_weights(newnet->in_layerpoint->input_weights, n_in, a[0]);
//#endif
	for (int i = 0; i < n_hiddenlayers; i++) {
		if (i != n_hiddenlayers - 1) {
			dpnn_randomize_weights(newnet->hidden_layerlist[i]->hidden_weights, a[i], a[i + 1]);
			dpnn_zero_weights(newnet->hidden_layerlist[i]->hidden_prev_weights, a[i], a[i + 1]);
		}
		else {
			dpnn_randomize_weights(newnet->hidden_layerlist[i]->hidden_weights, a[i], n_out);
			dpnn_zero_weights(newnet->hidden_layerlist[i]->hidden_prev_weights, a[i], n_out);
		}
	}

	dpnn_zero_weights(newnet->in_layerpoint->input_prev_weights, n_in, a[0]);

	return newnet;
}


/*void bpnn_layerforward(double *l1, double *l2, double **conn, int n1, int n2)
{
  double sum;
  int j, k;

  // Set up thresholding unit *
  l1[0] = 1.0;

  // For each unit in second layer **
  for (j = 1; j <= n2; j++) {

    // Compute weighted sum of its inputs **
    sum = 0.0;
    for (k = 0; k <= n1; k++) {
      sum += conn[k][j] * l1[k];
    }
    l2[j] = squash(sum);
  }

}*/

void dpnn_layerforward(double *l1, double *l2, double **conn, int n1, int n2)
{
	double sum;
	int j, k;

	/*** Set up thresholding unit ***/
	l1[0] = 1.0;

	/*** For each unit in second layer ***/
	for (j = 1; j <= n2; j++) {

		/*** Compute weighted sum of its inputs ***/
		sum = 0.0;
		for (k = 0; k <= n1; k++) {
			sum += conn[k][j] * l1[k];
		}
		l2[j] = squash(sum);
	}

}

/*void bpnn_output_error(double *delta, double *target, double *output, int nj, double *err)
{
  int j;
  double o, t, errsum;

  errsum = 0.0;
  for (j = 1; j <= nj; j++) {
    o = output[j];
    t = target[j];
    delta[j] = o * (1.0 - o) * (t - o);
    errsum += ABS(delta[j]);
  }
  *err = errsum;
}*/

void dpnn_output_error(double *delta, double *target, double *output, int nj, double *err)
{
	int j;
	double o, t, errsum;

	errsum = 0.0;
	for (j = 1; j <= nj; j++) {
		o = output[j];
		t = target[j];
		delta[j] = o * (1.0 - o) * (t - o);
		errsum += ABS(delta[j]);
	}
	*err = errsum;
}



/*void bpnn_hidden_error(double *delta_h, int nh, double *delta_o, int no, double **who, double *hidden, double *err)
{
  int j, k;
  double h, sum, errsum;

  errsum = 0.0;
  for (j = 1; j <= nh; j++) {
    h = hidden[j];
    sum = 0.0;
    for (k = 1; k <= no; k++) {
      sum += delta_o[k] * who[j][k];
    }
    delta_h[j] = h * (1.0 - h) * sum;
    errsum += ABS(delta_h[j]);
  }
  *err = errsum;
}*/
/*void bpnn_hidden_error_multihiddenlevel(double *delta_j, int nj, double *delta_k, int nk, double **connect, double *hidden_j, double *err)
{
	int j, k;
	double z, sum, errsum;

	errsum = 0.0;
	for (j = 1; j <= nj; j++) {
		z = hidden_j[j];
		sum = 0.0;
		for (k = 1; k <= nk; k++) {
			sum += delta_k[k] * connect[j][k];
		}
		delta_j[j] = z * (1.0 - z) * sum;
		errsum += ABS(delta_j[j]);
	}
	*err = errsum;
}*/
void dpnn_hidden_error(double *delta_j, int nj, double *delta_k, int nk, double **connect, double *hidden_j, double *err)
{
	int j, k;
	double z, sum, errsum;

	errsum = 0.0;
	for (j = 1; j <= nj; j++) {
		z = hidden_j[j];
		sum = 0.0;
		for (k = 1; k <= nk; k++) {
			sum += delta_k[k] * connect[j][k];
		}
		delta_j[j] = z * (1.0 - z) * sum;
		errsum += ABS(delta_j[j]);
	}
	*err = errsum;
}




/*void bpnn_adjust_weights(double *delta, int ndelta, double *ly, int nly, double **w, double **oldw, double eta, double momentum)
{
  double new_dw;
  int k, j;

  ly[0] = 1.0;
  for (j = 1; j <= ndelta; j++) {
    for (k = 0; k <= nly; k++) {
      new_dw = ((eta * delta[j] * ly[k]) + (momentum * oldw[k][j]));
      w[k][j] += new_dw;
      oldw[k][j] = new_dw;
    }
  }
}*/
/*void bpnn_adjust_weights_multihiddenlevel(double *delta_j, int nj, double *z_i, int ni, double **w, double **oldw, double learn_rate, double momentum)
{
	double new_dw;
	int i, j;

	z_i[0] = 1.0;
	for (j = 1; j <= nj; j++) {
		for (i = 0; i <= ni; i ++) {
			new_dw = ((learn_rate * delta_j[j] *z_i[i]) + (momentum * oldw[i][j]));
			w[i][j] += new_dw;
			oldw[i][j] = new_dw;
		}
	}
}*/
void dpnn_adjust_weights(double *delta_j, int nj, double *z_i, int ni, double **w, double **oldw, double learn_rate, double momentum)
{
	double new_dw;
	int i, j;

	z_i[0] = 1.0;
	for (j = 1; j <= nj; j++) {
		for (i = 0; i <= ni; i++) {
			new_dw = ((learn_rate * delta_j[j] * z_i[i]) + (momentum * oldw[i][j]));
			w[i][j] += new_dw;
			oldw[i][j] = new_dw;
		}
	}
}



/*void bpnn_feedforward(BPNN *net)
{
  int in, hid, out;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

  
  bpnn_layerforward(net->input_units, net->hidden_units,
      net->input_weights, in, hid);
  bpnn_layerforward(net->hidden_units, net->output_units,
      net->hidden_weights, hid, out);

}*/
/*void bpnn_feedforward_multihiddenlevel(BPNN *net)
{
	int in, hid_l1, hid_l2, hid_l3, out;

	in = net->input_n;
	hid_l1 = net->hidden_l1n;
	hid_l2 = net->hidden_l2n;
	hid_l3 = net->hidden_l3n;
	out = net->output_n;

	// Feed forward input activations. ***
	bpnn_layerforward(net->input_units, net->hidden_l1_units,
		net->input_weights, in, hid_l1);
	bpnn_layerforward(net->hidden_l1_units, net->hidden_l2_units,
		net->hidden_l1_weights, hid_l1, hid_l2);
	bpnn_layerforward(net->hidden_l2_units, net->hidden_l3_units,
		net->hidden_l2_weights, hid_l2, hid_l3);
	bpnn_layerforward(net->hidden_l3_units, net->output_units,
		net->hidden_l3_weights, hid_l3, out);

}*/

void dpnn_feedforward(DPNN *net)
{
	int in, hid_n, out;


	in = net->in_layerpoint->input_n;
	hid_n = net->hiddenlayer_n;
	out = net->out_layerpoint->output_n;

	/*** Feed forward input activations. ***/
	dpnn_layerforward(net->in_layerpoint->input_units, net->hidden_layerlist[0]->hidden_units,
		net->in_layerpoint->input_weights, in, net->hidden_layerlist[0]->hidden_n);
	for (int i = 0; i < hid_n; i++) {
		if (i != hid_n - 1) {
			dpnn_layerforward(net->hidden_layerlist[i]->hidden_units, net->hidden_layerlist[i + 1]->hidden_units,
				net->hidden_layerlist[i]->hidden_weights, net->hidden_layerlist[i]->hidden_n, net->hidden_layerlist[i + 1]->hidden_n);
		}
		else
		{
			dpnn_layerforward(net->hidden_layerlist[i]->hidden_units, net->out_layerpoint->output_units,
				net->hidden_layerlist[i]->hidden_weights, net->hidden_layerlist[i]->hidden_n, net->out_layerpoint->output_n);
		}

	}
}



/*void bpnn_train(BPNN *net,double eta, double momentum, double *eo, double *eh)
{
  int in, hid, out;
  double out_err, hid_err;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

    // Feed forward input activations. 
  bpnn_layerforward(net->input_units, net->hidden_units,
      net->input_weights, in, hid);
  bpnn_layerforward(net->hidden_units, net->output_units,
      net->hidden_weights, hid, out);

   // Compute error on output and hidden units. 
  bpnn_output_error(net->output_delta, net->target, net->output_units,
      out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out,
      net->hidden_weights, net->hidden_units, &hid_err);
  *eo = out_err;
  *eh = hid_err;

  // Adjust input and hidden weights. 
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid,
      net->hidden_weights, net->hidden_prev_weights, eta, momentum);
  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in,
      net->input_weights, net->input_prev_weights, eta, momentum);

}*/

/*void bpnn_train_multihiddenlevel(BPNN *net, double eta, double momentum, double *eo, double *eh_l1, double *eh_l2, double *eh_l3 )
{
	int in, hid_l1, hid_l2, hid_l3, out;
	double out_err, hid_l1_err , hid_l2_err, hid_l3_err;

	in = net->input_n;
	hid_l1 = net->hidden_l1n;
	hid_l2 = net->hidden_l2n;
	hid_l3 = net->hidden_l3n;
	out = net->output_n;

	// Feed forward input activations. ***
	bpnn_layerforward(net->input_units, net->hidden_l1_units,
		net->input_weights, in, hid_l1);
	bpnn_layerforward(net->hidden_l1_units, net->hidden_l2_units,
		net->hidden_l1_weights, hid_l1, hid_l2);
	bpnn_layerforward(net->hidden_l2_units, net->hidden_l3_units,
		net->hidden_l2_weights, hid_l2, hid_l3);
	bpnn_layerforward(net->hidden_l3_units, net->output_units,
		net->hidden_l3_weights, hid_l3, out);

	/*** Compute error on output and hidden units. ***
	bpnn_output_error(net->output_delta, net->target, net->output_units,
		out, &out_err);
	bpnn_hidden_error_multihiddenlevel(net->hidden_l3_delta, hid_l3, net->output_delta, out,
		net->hidden_l3_weights, net->hidden_l3_units, &hid_l3_err);
	bpnn_hidden_error_multihiddenlevel(net->hidden_l2_delta, hid_l2, net->hidden_l3_delta, hid_l3,
		net->hidden_l2_weights, net->hidden_l2_units, &hid_l2_err);
	bpnn_hidden_error_multihiddenlevel(net->hidden_l1_delta, hid_l1, net->hidden_l2_delta, hid_l2,
		net->hidden_l1_weights, net->hidden_l1_units, &hid_l1_err);
	*eo = out_err;
	*eh_l1 = hid_l1_err;
	*eh_l2 = hid_l2_err;
	*eh_l3 = hid_l3_err;

	/*** Adjust input and hidden weights. ***
	bpnn_adjust_weights_multihiddenlevel(net->output_delta, out, net->hidden_l3_units, hid_l3,
		net->hidden_l3_weights, net->hidden_l3_prev_weights, eta, momentum);
	bpnn_adjust_weights_multihiddenlevel(net->hidden_l3_delta, hid_l3, net->hidden_l2_units, hid_l2,
		net->hidden_l2_weights, net->hidden_l2_prev_weights, eta, momentum);
	bpnn_adjust_weights_multihiddenlevel(net->hidden_l2_delta, hid_l2, net->hidden_l1_units, hid_l1,
		net->hidden_l1_weights, net->hidden_l1_prev_weights, eta, momentum);
	bpnn_adjust_weights_multihiddenlevel(net->hidden_l1_delta, hid_l1, net->input_units, in,
		net->input_weights, net->input_prev_weights, eta, momentum);

}*/

void dpnn_train(DPNN *net, double eta, double momentum, double *eo, double *ehidden)
{
	int in, hid_n, out;
	double out_err;


	in = net->in_layerpoint->input_n;
	out = net->out_layerpoint->output_n;
	hid_n = net->hiddenlayer_n;


	/*** Feed forward input activations. ***/
	dpnn_layerforward(net->in_layerpoint->input_units, net->hidden_layerlist[0]->hidden_units,
		net->in_layerpoint->input_weights, in, net->hidden_layerlist[0]->hidden_n);
	for (int i = 0; i < hid_n; i++) {
		if (i != hid_n - 1) {
			dpnn_layerforward(net->hidden_layerlist[i]->hidden_units, net->hidden_layerlist[i + 1]->hidden_units,
				net->hidden_layerlist[i]->hidden_weights, net->hidden_layerlist[i]->hidden_n, net->hidden_layerlist[i + 1]->hidden_n);
		}
		else
		{
			dpnn_layerforward(net->hidden_layerlist[i]->hidden_units, net->out_layerpoint->output_units,
				net->hidden_layerlist[i]->hidden_weights, net->hidden_layerlist[i]->hidden_n, net->out_layerpoint->output_n);
		}

	}

	/*** Compute error on output and hidden units. ***/
	dpnn_output_error(net->out_layerpoint->output_delta, net->target, net->out_layerpoint->output_units,
		out, &out_err);
	dpnn_hidden_error(net->hidden_layerlist[hid_n - 1]->hidden_delta, net->hidden_layerlist[hid_n-1]->hidden_n,
		net->out_layerpoint->output_delta, out, net->hidden_layerlist[hid_n - 1]->hidden_weights, net->hidden_layerlist[hid_n-1]->hidden_units, &ehidden[hid_n-1]);
	for (int i = hid_n-1; i > 0; i--) {
		dpnn_hidden_error(net->hidden_layerlist[i - 1]->hidden_delta, net->hidden_layerlist[i - 1]->hidden_n,
				net->hidden_layerlist[i]->hidden_delta, net->hidden_layerlist[i]->hidden_n, net->hidden_layerlist[i - 1]->hidden_weights, net->hidden_layerlist[i - 1]->hidden_units, &ehidden[i - 1]);
	}
	*eo = out_err;

	/*** Adjust input and hidden weights. ***/
	dpnn_adjust_weights(net->out_layerpoint->output_delta, out, net->hidden_layerlist[hid_n-1]->hidden_units, net->hidden_layerlist[hid_n-1]->hidden_n,
		net->hidden_layerlist[hid_n-1]->hidden_weights, net->hidden_layerlist[hid_n-1]->hidden_prev_weights, eta, momentum);
	for (int i = hid_n - 1; i > 0; i--) {
		dpnn_adjust_weights(net->hidden_layerlist[i]->hidden_delta, net->hidden_layerlist[i]->hidden_n, net->hidden_layerlist[i - 1]->hidden_units, net->hidden_layerlist[i - 1]->hidden_n,
			net->hidden_layerlist[i - 1]->hidden_weights, net->hidden_layerlist[i - 1]->hidden_prev_weights, eta, momentum);
	}
	dpnn_adjust_weights(net->hidden_layerlist[0]->hidden_delta, net->hidden_layerlist[0]->hidden_n, net->in_layerpoint->input_units, net->in_layerpoint->input_n,
		net->in_layerpoint->input_weights, net->in_layerpoint->input_prev_weights, eta, momentum);

}





/*void bpnn_save(BPNN *net,char *filename)
{
  int n1, n2, n3, i, j, memcnt;
  double dvalue, **w;
  char *mem;
  FILE* fd;

  if ((fd = fopen(filename, "wb")) == NULL) {
    printf("BPNN_SAVE: Cannot create '%s'\n", filename);
    return;
  }

  n1 = net->input_n;  n2 = net->hidden_n;  n3 = net->output_n;
  printf("Saving %dx%dx%d network to '%s'\n", n1, n2, n3, filename);
  fflush(stdout);

  fwrite( (char *) &n1, sizeof(int), 1, fd);
  fwrite( (char *) &n2, sizeof(int), 1, fd);
  fwrite( (char *) &n3, sizeof(int), 1, fd);

  memcnt = 0;
  w = net->input_weights;
  mem = (char *) malloc ((unsigned) ((n1+1) * (n2+1) * sizeof(double)));
  for (i = 0; i <= n1; i++) {
    for (j = 0; j <= n2; j++) {
      dvalue = w[i][j];
      fastcopy(&mem[memcnt], &dvalue, sizeof(double));
      memcnt += sizeof(double);
    }
  }
  fwrite(mem, (n1+1) * (n2+1) * sizeof(double), 1, fd);
  free(mem);

  memcnt = 0;
  w = net->hidden_weights;
  mem = (char *) malloc ((unsigned) ((n2+1) * (n3+1) * sizeof(double)));
  for (i = 0; i <= n2; i++) {
    for (j = 0; j <= n3; j++) {
      dvalue = w[i][j];
      fastcopy(&mem[memcnt], &dvalue, sizeof(double));
      memcnt += sizeof(double);
    }
  }
  fwrite( mem, (n2+1) * (n3+1) * sizeof(double), 1, fd );
  free(mem);

  fclose(fd);
  return;
}*/

void dpnn_save(DPNN *net, char *filename)
{
	int n_in, hid_n, n_out, i, j, memcnt;
	double dvalue, **w;
	char *mem;
	FILE* fd;

	if ((fd = fopen(filename, "wb")) == NULL) {
		printf("BPNN_SAVE: Cannot create '%s'\n", filename);
		return;
	}

	n_in = net->in_layerpoint->input_n;  
	n_out = net->out_layerpoint->output_n;
	hid_n = net->hiddenlayer_n;

	printf("Saving %dx%d (hidden layers) x%d network to '%s'\n", n_in, hid_n, n_out, filename);
	fflush(stdout);

	fwrite((char *)&n_in, sizeof(int), 1, fd);
	fwrite((char *)&n_out, sizeof(int), 1, fd);
	fwrite((char *)&hid_n, sizeof(int), 1, fd);
	for (int i = 0; i < hid_n; i++) {
		fwrite((char *)&net->hidden_layerlist[i]->hidden_n, sizeof(int), 1, fd);
	}

	memcnt = 0;
	w = net->in_layerpoint->input_weights;
	mem = (char *)malloc((unsigned)((n_in + 1) * (net->hidden_layerlist[0]->hidden_n + 1) * sizeof(double)));
	for (i = 0; i <= n_in; i++) {
		for (j = 0; j <= net->hidden_layerlist[0]->hidden_n; j++) {
			dvalue = w[i][j];
			fastcopy(&mem[memcnt], &dvalue, sizeof(double));
			memcnt += sizeof(double);
		}
	}
	fwrite(mem, (n_in + 1) * (net->hidden_layerlist[0]->hidden_n + 1) * sizeof(double), 1, fd);
	free(mem);

	for (int k = 0; k < hid_n-1; k++) {
		memcnt = 0;
		w = net->hidden_layerlist[k]->hidden_weights;
		mem = (char *)malloc((unsigned)((net->hidden_layerlist[k]->hidden_n + 1) * (net->hidden_layerlist[k+1]->hidden_n + 1) * sizeof(double)));
		for (i = 0; i <= net->hidden_layerlist[k]->hidden_n ; i++) {
			for (j = 0; j <= net->hidden_layerlist[k + 1]->hidden_n ; j++) {
				dvalue = w[i][j];
				fastcopy(&mem[memcnt], &dvalue, sizeof(double));
				memcnt += sizeof(double);
			}
		}
		fwrite(mem, (net->hidden_layerlist[k]->hidden_n + 1) * (net->hidden_layerlist[k + 1]->hidden_n + 1) * sizeof(double), 1, fd);
		free(mem);
	}
	memcnt = 0;
	w = net->hidden_layerlist[hid_n-1]->hidden_weights;
	mem = (char *)malloc((unsigned)((net->hidden_layerlist[hid_n-1]->hidden_n + 1) * (net->out_layerpoint->output_n + 1) * sizeof(double)));
	for (i = 0; i <= net->hidden_layerlist[hid_n-1]->hidden_n; i++) {
		for (j = 0; j <= net->out_layerpoint->output_n; j++) {
			dvalue = w[i][j];
			fastcopy(&mem[memcnt], &dvalue, sizeof(double));
			memcnt += sizeof(double);
		}
	}
	fwrite(mem, (net->hidden_layerlist[hid_n-1]->hidden_n + 1) * (net->out_layerpoint->output_n + 1) * sizeof(double), 1, fd);
	free(mem);

	fclose(fd);
	return;
}


/*BPNN *bpnn_read(char *filename)
{
  char *mem;
  BPNN *newptr;
  int n1, n2, n3, i, j, memcnt;
  FILE *fd;

  if ((fd = fopen(filename, "rb")) == NULL) {
    return (NULL);
  }

  printf("Reading '%s'\n", filename);  fflush(stdout);

  fread((char *) &n1, sizeof(int), 1, fd );
  fread((char *) &n2, sizeof(int), 1, fd );
  fread((char *) &n3, sizeof(int), 1, fd );
  newptr = bpnn_internal_create(n1, n2, n3);

  printf("'%s' contains a %dx%dx%d network\n", filename, n1, n2, n3);
  printf("Reading input weights...");  fflush(stdout);

  memcnt = 0;
  mem = (char *) malloc ((unsigned) ((n1+1) * (n2+1) * sizeof(double)));
  fread( mem, (n1+1) * (n2+1) * sizeof(double), 1, fd );
  for (i = 0; i <= n1; i++) {
    for (j = 0; j <= n2; j++) {
      fastcopy(&(newptr->input_weights[i][j]), &mem[memcnt], sizeof(double));
      memcnt += sizeof(double);
    }
  }
  free(mem);

  printf("Done\nReading hidden weights...");  fflush(stdout);

  memcnt = 0;
  mem = (char *) malloc ((unsigned) ((n2+1) * (n3+1) * sizeof(double)));
  fread( mem, (n2+1) * (n3+1) * sizeof(double), 1, fd );
  for (i = 0; i <= n2; i++) {
    for (j = 0; j <= n3; j++) {
      fastcopy(&(newptr->hidden_weights[i][j]), &mem[memcnt], sizeof(double));
      memcnt += sizeof(double);
    }
  }
  free(mem);
  fclose(fd);

  printf("Done\n");  fflush(stdout);

  bpnn_zero_weights(newptr->input_prev_weights, n1, n2);
  bpnn_zero_weights(newptr->hidden_prev_weights, n2, n3);

  return (newptr);
}*/

DPNN *dpnn_read(char *filename)
{
	char *mem;
	DPNN *newptr;
	int n1, hid_n, n3, i, j, memcnt;
	FILE *fd;
	

	if ((fd = fopen(filename, "rb")) == NULL) {
		return (NULL);
	}

	printf("Reading '%s'\n", filename);  fflush(stdout);

	fread((char *)&n1, sizeof(int), 1, fd);
	fread((char *)&n3, sizeof(int), 1, fd);
	fread((char *)&hid_n, sizeof(int), 1, fd);
	int *a = (int *)malloc((unsigned) hid_n * sizeof(int));
	for (int i = 0; i < hid_n; i++) {
		fread((char *)&a[i], sizeof(int), 1, fd);
	}
	newptr = dpnn_internal_create_net(n1,hid_n,n3,a);

	printf("'%s' contains a %dx%d (hidden layers)x%d network\n", filename, n1 , hid_n, n3);
	printf("Reading input weights...");  fflush(stdout);

	memcnt = 0;
	mem = (char *)malloc((unsigned)((n1 + 1) * (a[0] + 1) * sizeof(double)));
	fread(mem, (n1 + 1) * (a[0] + 1) * sizeof(double), 1, fd);
	for (i = 0; i <= n1; i++) {
		for (j = 0; j <= a[0]; j++) {
			fastcopy(&(newptr->in_layerpoint->input_weights[i][j]), &mem[memcnt], sizeof(double));
			memcnt += sizeof(double);
		}
	}
	free(mem);

	printf("Done\nReading hidden weights...");  fflush(stdout);
	for (int k = 0; k < hid_n-1; k++) {
		memcnt = 0;
		mem = (char *)malloc((unsigned) ((a[k] + 1) * (a[k + 1] + 1) * sizeof(double)));
		fread(mem, (a[k] + 1) * (a[k + 1] + 1) * sizeof(double), 1, fd);
		for (i = 0; i <= a[i]; i++) {
			for (j = 0; j <= a[i + 1]; j++) {
				fastcopy(&(newptr->hidden_layerlist[k]->hidden_weights[i][j]), &mem[memcnt], sizeof(double));
				memcnt += sizeof(double);
			}
		}
		free(mem);
	}
	memcnt = 0;
	mem = (char *)malloc((unsigned)((a[hid_n-1] + 1) * (n3 + 1) * sizeof(double)));
	fread(mem, (a[hid_n-1] + 1) * (n3 + 1) * sizeof(double), 1, fd);
	for (i = 0; i <= a[hid_n - 1]; i++) {
		for (j = 0; j <= n3; j++) {
			fastcopy(&(newptr->hidden_layerlist[hid_n - 1]->hidden_weights[i][j]), &mem[memcnt], sizeof(double));
			memcnt += sizeof(double);
		}
	}
	free(mem);

	fclose(fd);
	printf("Done\n");  fflush(stdout);

	dpnn_zero_weights(newptr->in_layerpoint->input_prev_weights, n1, a[0]);
	for (int i = 0; i < hid_n-1; i++) {
		dpnn_zero_weights(newptr->hidden_layerlist[i]->hidden_prev_weights, a[i], a[i+1]);
	}
	dpnn_zero_weights(newptr->hidden_layerlist[hid_n-1]->hidden_prev_weights, a[hid_n-1],n3 );

	return (newptr);
}

