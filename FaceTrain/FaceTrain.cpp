// FaceTrain.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "FaceTrain.h"
#include <fstream>
#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// The one and only application object

CWinApp theApp;


/*
******************************************************************
* HISTORY
* 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
*      Prepared for 15-681, Fall 1994.
*
* Tue Oct  7 08:12:06 EDT 1997, bthom, added a few comments,
*       tagged w/bthom
*
******************************************************************
*/

#include <stdio.h>
#include <math.h>
#include "pgmimage.h"
#include "backprop.h"

extern char *strcpy();
extern void exit();
void printusage(char *prog);
void backprop_face(IMAGELIST *trainlist, IMAGELIST *test1list, IMAGELIST *test2list, int epochs, int savedelta, char* netname,
				   int list_errors);
void performance_on_imagelist(DPNN *net,IMAGELIST *il,int list_errors,std::ofstream * out);
int evaluate_performance(DPNN *net, double *err);
int find_max(double * l);
double sum(double* a, int n);
void load_input_with_image(IMAGE *img,DPNN *net);
void load_target(IMAGE *img,DPNN *net);


void backprop_face(IMAGELIST *trainlist, IMAGELIST *test1list, IMAGELIST *test2list, int epochs, int savedelta, char* netname,
			  int list_errors)
{
	IMAGE *iimg;
	DPNN *net;
	int train_n, epoch, i, imgsize;
	double out_err, *hidden_errors, sumerr;
    

	int hidden_points[2];
	hidden_points[0] = 20;
	hidden_points[1] = 10;


	int hidden_layer_n = sizeof(hidden_points) / sizeof(hidden_points[0]);
	hidden_errors = (double *)malloc((unsigned) hidden_layer_n * (sizeof (double)));

	train_n = trainlist->n;

	/*** Read network in if it exists, otherwise make one from scratch ***/
	if ((net = dpnn_read(netname)) == NULL) {
		if (train_n > 0) {
			printf("Creating new network '%s'\n", netname);
			iimg = trainlist->list[0];
			imgsize = ROWS(iimg) * COLS(iimg);


			/* bthom ===========================
			make a net with:
			imgsize inputs, 4 hiden units, and 1 output unit
			*/
			net = dpnn_create(imgsize, hidden_layer_n, 4, hidden_points);
		

		} else {
			printf("Need some images to train on, use -t\n");
			return;
		}
	}

	if (epochs > 0) {
		printf("Training underway (going to %d epochs)\n", epochs);
		printf("Will save network every %d epochs\n", savedelta);
		fflush(stdout);
	}

	/*** Print out performance before any epochs have been completed. ***/
	printf("0 0.0 ");

	std::ofstream out;
	out.open("output.txt");

	performance_on_imagelist(net, trainlist, 0, &out);
	performance_on_imagelist(net, test1list, 0, &out);
	performance_on_imagelist(net, test2list, 0, &out);
	printf("\n");  fflush(stdout);
	if (list_errors) {
		printf("\nFailed to classify the following images from the training set:\n");
		performance_on_imagelist(net, trainlist, 1, &out);
		printf("\nFailed to classify the following images from the test set 1:\n");
		performance_on_imagelist(net, test1list, 1, &out);
		printf("\nFailed to classify the following images from the test set 2:\n");
		performance_on_imagelist(net, test2list, 1, &out);
	}

	/************** Train it *****************************/
	for (epoch = 1; epoch <= epochs; epoch++) {

		printf("epoch: %d ", epoch);  fflush(stdout);

		sumerr = 0.0;
		for (i = 0; i < train_n; i++) {

			/** Set up input units on net with image i **/
			load_input_with_image(trainlist->list[i], net);

			/** Set up target vector for image i **/
			load_target(trainlist->list[i], net);

			/** Run backprop, learning rate 0.3, momentum 0.3 **/
			dpnn_train(net, 0.05, 0.1, &out_err, hidden_errors);
			//printf("hidden_ers:%f,%f;", hidden_errors[0], hidden_errors[1]);
			sumerr += (out_err + sum(hidden_errors, hidden_layer_n));

		}
		printf("sumerr:%g ", sumerr);
		
		/*** Evaluate performance on train, test, test2, and print perf ***/
		out << epoch << ",";
		out << sumerr << ",";
		performance_on_imagelist(net, trainlist, 0, &out);
		out << ",";
		performance_on_imagelist(net, test1list, 0, &out);
		out << ",";
		performance_on_imagelist(net, test2list, 0, &out);
		out << std::endl;
		printf("\n");  fflush(stdout);

		/*** Save network every 'savedelta' epochs ***/
		if (!(epoch % savedelta)) {
			dpnn_save(net, netname);
		}
	}
	printf("\n"); fflush(stdout);

	/** Save the trained network **/
	if (epochs > 0) {
		dpnn_save(net, netname);
	}
	out.close();
}
int find_max(double * l){
	int max = 0;
	double max_value = 0;
	for (int i = 1;i < 5;i++){
    	if (l[i] > max_value){
			max_value = l[i];
    		max = i;
    	}
  	}
  	return max;
}
int evaluate_performance(DPNN *net,double *err)
{
	double delta = 0;

	for (int i = 1;i < 5;i++){
		delta += (net->target[i] -net->out_layerpoint->output_units[i]) * (net->target[i] - net->out_layerpoint->output_units[i]);
  	}

	*err = delta;

	if (find_max(net->target) == find_max(net->out_layerpoint->output_units)){
		return 1;
	} else {
		return 0;
	}
}
 
double sum(double* a, int n) {
	double tmp = 0;
	for (int i = 0; i < n; i++) {
		tmp += a[i];
	}
	return tmp;
}


/*** Computes the performance of a net on the images in the imagelist. ***/
/*** Prints out the percentage correct on the image set, and the
average error between the target and the output units for the set. ***/
void performance_on_imagelist(DPNN *net, IMAGELIST *il, int list_errors, std::ofstream * out)
{
	double err, val;
	int i, n, j, correct;

	err = 0.0;
	correct = 0;
	n = il->n;
	if (n > 0) {
		for (i = 0; i < n; i++) {
			//printf("%d\n",i);

			/*** Load the image into the input layer. **/
			load_input_with_image(il->list[i], net);

			/*** Run the net on this input. **/
			dpnn_feedforward(net);

			/*** Set up the target vector for this image. **/
			load_target(il->list[i], net);
			//printf("target:%f,%f,%f,%f\n", net->target[1], net->target[2], net->target[3], net->target[4]);
			/*** See if it got it right. ***/
			//chaged by sjt here.
			//if (evaluate_performance(net, &val, 0)) {
			if (evaluate_performance(net, &val)) {
				correct++;
			} else if (list_errors) {
				printf("%s - outputs ", NAME(il->list[i]));
				for (j = 1; j <= net->out_layerpoint->output_n; j++) {
					printf("%.3f ", net->out_layerpoint->output_units[j]);
				}
				putchar('\n');
			}
			err += val;
		}

		err = err / (double) (2*n);

		if (!list_errors){
			/* bthom==================================
			this line prints part of the ouput line
			discussed in section 3.1.2 of homework
			*/
			printf("perf:%g ,err:%g ", ((double) correct / (double) n) * 100.0, err);
			*out << ((double) correct / (double) n) << "," << err;
		}
	} else {
		if (!list_errors)
			printf("0.0 0.0 ");
	}
}


void printusage(char *prog)
{
	printf("USAGE: %s\n", prog);
	printf("       -n <network file>\n");
	printf("       [-e <number of epochs>]\n");
	printf("       [-s <random number generator seed>]\n");
	printf("       [-S <number of epochs between saves of network>]\n");
	printf("       [-t <training set list>]\n");
	printf("       [-1 <testing set 1 list>]\n");
	printf("       [-2 <testing set 2 list>]\n");
	printf("       [-T]\n");
}


using namespace std;

int _tmain(int argc, TCHAR* argv[], TCHAR* envp[])
{
	int nRetCode = 0;
	printf("exe.weizhi : %s\n", argv[0]);
	// initialize MFC and print and error on failure
	if (!AfxWinInit(::GetModuleHandle(NULL), NULL, ::GetCommandLine(), 0))
	{
		// TODO: change error code to suit your needs
		cerr << _T("Fatal Error: MFC initialization failed") << endl;
		nRetCode = 1;
	}

    
	char netname[256], trainname[256], test1name[256], test2name[256];
	IMAGELIST *trainlist, *test1list, *test2list;
	int ind, epochs, seed, savedelta, list_errors;

	seed = 102194;   /*** today's date seemed like a good default ***/
	epochs = 100;
	savedelta = 100;
	list_errors = 0;
	netname[0] = trainname[0] = test1name[0] = test2name[0] = '\0';

	if (argc < 2) {
		printusage(argv[0]);
		exit (-1);
	}

	/*** Create imagelists ***/
	trainlist = imgl_alloc();
	test1list = imgl_alloc();
	test2list = imgl_alloc();

	/*** Scan command line ***/
	for (ind = 1; ind < argc; ind++) {

		/*** Parse switches ***/
		if (argv[ind][0] == '-') {
			switch (argv[ind][1]) {  
		case 'n': strcpy(netname, argv[++ind]);
			break;
		case 'e': epochs = atoi(argv[++ind]);
			break;
		case 's': seed = atoi(argv[++ind]);
			break;
		case 'S': savedelta = atoi(argv[++ind]);
			break;
		case 't': strcpy(trainname, argv[++ind]);
			break;
		case '1': strcpy(test1name, argv[++ind]);
			break;
		case '2': strcpy(test2name, argv[++ind]);
			break;
		case 'T': list_errors = 1;
			epochs = 0;
			break;
		default : printf("Unknown switch '%c'\n", argv[ind][1]);
			break;
			}
		}
	}

	/*** If any train, test1, or test2 sets have been specified, then
	load them in. ***/
	if (trainname[0] != '\0') 
		imgl_load_images_from_textfile(trainlist, trainname);
	if (test1name[0] != '\0') 
		imgl_load_images_from_textfile(test1list, test1name);
	if (test2name[0] != '\0')
		imgl_load_images_from_textfile(test2list, test2name);

	/*** If we haven't specified a network save file, we should... ***/
	if (netname[0] == '\0') {
		printf("%s: Must specify an output file, i.e., -n <network file>\n",
			argv[0]);
		exit (-1);
	}

	/*** Don't try to train if there's no training data ***/
	if (trainname[0] == '\0') {
		epochs = 0;
	}

	/*** Initialize the neural net package ***/
	dpnn_initialize(seed);

	/*** Show number of images in train, test1, test2 ***/
	printf("%d images in training set\n", trainlist->n);
	printf("%d images in test1 set\n", test1list->n);
	printf("%d images in test2 set\n", test2list->n);

	/*** If we've got at least one image to train on, go train the net ***/
	backprop_face(trainlist, test1list, test2list, epochs, savedelta, netname,
		list_errors);
	
	return nRetCode;

}


