#this is a module that contains tools for special types of network 
#the file yann.special.gan.py contains the definition for gan-style-network. Any GAN network can be built using this class
from yann.special.gan import gan 
from theano import tensor as T 

def shallow_gan_mnist ( dataset = None, verbose = 1 ):
    """
    This function is an example of a generative adversarial network. 
    Args: 
        dataset: Supply a dataset.    
        verbose: Similar to the rest of the dataset.
    Notes:
        This method is setup for MNIST.
    """


    optimizer_params =  {        
                "momentum_type"       : 'polyak',             
                "momentum_params"     : (0.65, 0.9, 50),      
                "regularization"      : (0.000, 0.000),       
                "optimizer_type"      : 'rmsprop',                
                "id"                  : "main"
                        }

  """ 
  Optimizers:
  creates the protocol required for learning. 
  Parameters:
      verbose and optimizer_init_args is of the form

  optimizer_params =  {
    "momentum_type"   : <option>  'false' <no momentum>, 'polyak', 'nesterov'.
                        Default value is 'false'
    "momentum_params" : (<option in range [0,1]>, <option in range [0,1]>, <int>)
                        (momentum coeffient at start,at end,
                        at what epoch to end momentum increase)
                        Default is the tuple (0.5, 0.95,50)
    "optimizer_type" : <option>, 'sgd', 'adagrad', 'rmsprop', 'adam'.
                       Default is 'sgd'
    "id"        : id of the optimizer
            }

  returns : optimizer object


  """



    dataset_params  = {
                            "dataset"   : dataset,
                            "type"      : 'xy',
                            "id"        : 'data'
                    }


   """
    Dataset:
     This module initializes the dataset to the network class and provides all dataset related functionalities. 
     It also provides for dynamically loading and caching dataset batches. 

     Parameters:
       dataset_init_args is of the form 
      
    dataset_init_args = {
            "dataset":  <location>
            "svm"    :  False or True
                 ``svm`` if ``True``, a one-hot label set will also be setup.
            "n_classes": <int>
                ``n_classes`` if ``svm`` is ``True``, we need to know how
                 many ``n_classes`` are present.
            "id": id of the datastream
                       }
     returns: A dataset module object that has the details of loader and other things.


         
   """

    visualizer_params = {
                    "root"       : 'shallow_gan/',
                    "frequency"  : 1,
                    "sample_size": 225,
                    "rgb_filters": False,
                    "debug_functions" : False,
                    "debug_layers": True,  
                    "id"         : 'main'
                        }  

   """
    Visualizer:
       Visualizer saves down images to visualize. 
      The initilizer only initializes the directories for storing visuals. 
      Three types of visualizations are saved down:

           filters of each layer
           activations of each layer
           raw images to check the activations against

      Parameters:	
            verbose – Similar to any 3-level verbose in the toolbox.
            visualizer_init_args –
            visualer_params is a dictionary of the form:

          visualizer_init_args = {
                  	"root"       : <location to save the visualizations at>,
    			"frequency"  : <integer>, after how many epochs do you need to
                    			visualize. Default value is 1
    			"sample_size": <integer, prefer squares>, simply save down random
                    			images from the datasets saves down activations for the
                    			same images also. Default value is 16
    			"rgb_filters": <bool> flag. if True a 3D-RGB rendition of the CNN
                    			filters is rendered. Default value is False.
    			"debug_functions" : <bool> visualize train and test and other theano functions.
                        		default is False. Needs pydot and dv2viz to be installed.
    			"debug_layers" : <bool> Will print layer activities from input to that layer
                     			output. ( this is almost always useless because test debug
                     			function will combine all these layers and print directly.)
    			"id"         : id of the visualizer
                		}
		Returns: A visualizer object.

   """
                      
    # intitialize the network with a datastream, visualizer and an optimizer

    net = gan (     borrow = True,
                    verbose = verbose )     

    """
      add_module(type, params=None, verbose=2): used to add a module to net

            type: which module to add. Options are 'datastream', 'visualizer', 'optimizer' and 'resultor'
            params: dicitionary as used above
	    verbose: similar to the rest of the toolbox

    """                  
    
    
    net.add_module ( type = 'datastream', 
                     params = dataset_params,
                     verbose = verbose )    
    
    net.add_module ( type = 'visualizer',
                     params = visualizer_params,
                     verbose = verbose 
                    ) 

    
    net.add_module ( type = 'optimizer',
                     params = optimizer_params,
                     verbose = verbose 
                    ) 


    """
      add_layer()
   	Parameters:

          	type – <string> options include ‘input’ or ‘data’ 
		id – <string> how to identify the layer by. Default is just layer number that starts with 0.
		origin – id will use the output of that layer as input to the new layer. Default is the last layer created.
		 This variable for input type of layers is not a layer, but a datastream id. For merge layer, this is a tuple of two layer ids.
		verbose – similar to the rest of the toolbox.
		mean_subtract – if True we will subtract the mean from each image, else not.
		num_neurons – number of neurons in the layer
		dataset – <string> Location to the dataset. used when layer type is input.
		activation – String, takes options that are listed in activations Needed for layers that use activations. Some activations also take 				support parameters, for instance maxout takes maxout type and size, softmax takes an option temperature. 
			Refer to the module activations to know more.
		stride – tuple (int , int). Used as convolution stride. Default (1,1)
		batch_norm – If provided will be used, default is False.
		border_mode – Refer to border_mode variable in yann.core.conv, module conv
		pool_size – Subsample size, default is (1,1).
		pool_type – Refer to pool for details. {‘max’, ‘sum’, ‘mean’, ‘max_same_size’}
		learnable – Default is True, if True we backprop on that layer. If False Layer is obstinate.
		shape – tuple of shape to unflatten to ( height, width, channels ) in case layer was an unflatten layer
		input_params – Supply params or initializations from a pre-trained system.
		dropout_rate – If you want to dropout this layer’s output provide the output.
		regularize – True is you want to apply regularization, False if not.
		num_classes – int number of classes to classify.
		objective – objective provided by classifier nll-negative log likelihood, cce-categorical cross entropy, bce-binary cross entropy, 			hinge- hinge loss . For classifier layer.
		dataset_init_args – same as for the dataset module. In fact this argument is needed only when dataset module is not setup.
		datastream_id – When using input layer or during objective layer, use this to identify which datastream to take data from.
		regularizer – Default is (0.001, 0.001) coeffients for L1, L2 regulaizer coefficients.
		error – merge layers take an option called 'error' which can be None or others which are methods in yann.core.errors.
		angle – Takes value between [0,1] to capture the angle between [0,180] degrees Default is None. If None is specified, random number is 				generated from a uniform distriibution between 0 and 1.
		layer_type – If value supply, else it is default 'discriminator'. For other layers, if the layer class takes an argument type, supply 				that argument here as layer_type. merge layer for instance will use this arugment as its type argument.
     

    """

    #z - latent space created by random layer
    net.add_layer(      type = 'random',
                        id = 'z',
                        num_neurons = (100,32), 
                        distribution = 'normal',
                        mu = 0,
                        sigma = 1,
                        verbose = verbose)

    
    #x - inputs come from dataset 1 X 784
    net.add_layer ( type = "input",
                    id = "x",
                    verbose = verbose, 
                    datastream_origin = 'data',
                                                 
                    mean_subtract = False )

    net.add_layer ( type = "dot_product",
                    origin = "z",
                    id = "G(z)",
                    num_neurons = 784,
                    activation = 'tanh',
                    verbose = verbose
                    )  # This layer is the one that creates the images.
        
    #D(x) - Contains params theta_d creates features 1 X 800. 
    net.add_layer ( type = "dot_product",
                    id = "D(x)",
                    origin = "x",
                    num_neurons = 800,
                    activation = 'relu',
                    regularize = True,                                                         
                    verbose = verbose
                    )

    net.add_layer ( type = "dot_product",
                    id = "D(G(z))",
                    origin = "G(z)",
                    input_params = net.dropout_layers["D(x)"].params, 
                    num_neurons = 800,
                    activation = 'relu',
                    regularize = True,
                    verbose = verbose
                    )


    #C(D(x)) - This is the opposite of C(D(G(z))), real
    net.add_layer ( type = "dot_product",
                    id = "real",
                    origin = "D(x)",
                    num_neurons = 1,
                    activation = 'sigmoid',
                    verbose = verbose
                    )

    #C(D(G(z))) fake - the classifier for fake/real that always predicts fake 
    net.add_layer ( type = "dot_product",
                    id = "fake",
                    origin = "D(G(z))",
                    num_neurons = 1,
                    activation = 'sigmoid',
                    input_params = net.dropout_layers["real"].params, # Again share their parameters                    
                    verbose = verbose
                    )

    
    #C(D(x)) - This is the opposite of C(D(G(z))), real
    net.add_layer ( type = "classifier",
                    id = "softmax",
                    origin = "D(x)",
                    num_classes = 10,
                    activation = 'softmax',
                    verbose = verbose
                   )
    
    # objective layers 
    # discriminator objective 
    
    net.add_layer (type = "tensor",
                   input =  - 0.5 * T.mean(T.log(net.layers['real'].output)) - \
                                  0.5 * T.mean(T.log(1-net.layers['fake'].output)),
                    input_shape = (1,),
                    id = "discriminator_task"
                    )

    net.add_layer ( type = "objective",
                    id = "discriminator_obj",
                    origin = "discriminator_task",
                    layer_type = 'value',
                    objective = net.dropout_layers['discriminator_task'].output,
                    datastream_origin = 'data', 
                    verbose = verbose
                    )
    #generator objective 
    net.add_layer (type = "tensor",
                    input =  - 0.5 * T.mean(T.log(net.layers['fake'].output)),
                    input_shape = (1,),
                    id = "objective_task"
                    )
    net.add_layer ( type = "objective",
                    id = "generator_obj",
                    layer_type = 'value',
                    origin = "objective_task",
                    objective = net.dropout_layers['objective_task'].output,
                    datastream_origin = 'data', 
                    verbose = verbose
                    )   

    #softmax objective.    
    net.add_layer ( type = "objective",
                    id = "classifier_obj",
                    origin = "softmax",
                    objective = "nll",
                    layer_type = 'discriminator',
                    datastream_origin = 'data', 
                    verbose = verbose
                    )
    
    from yann.utils.graph import draw_network
        
    net.pretty_print()   #This method is used to pretty print the network’s connections 
    
    net.cook (  objective_layers = ["classifier_obj", "discriminator_obj", "generator_obj"],
                optimizer_params = optimizer_params,
                discriminator_layers = ["D(x)"],
                generator_layers = ["G(z)"], 
                classifier_layers = ["D(x)", "softmax"],                                                
                softmax_layer = "softmax",
                game_layers = ("fake", "real"),
                verbose = verbose )


   """
   This function builds the backprop network, and makes the trainer, tester and validator theano functions. 
   The trainer builds the trainers for a particular objective layer and optimizer.

	Parameters:	
		optimizer – Supply which optimizer to use. Default is last optimizer created.
		datastream – Supply which datastream to use. Default is the last datastream created.
		visualizer – Supply a visualizer to cook with. Default is the last visualizer created.
		classifier_layer – supply the layer of classifier. Default is the last classifier layer created.
		objective_layers – Supply a list of layer ids of layers that has the objective function. Default is last objective layer created if no 					classifier is provided.
		objective_weights – Supply a list of weights to be multiplied by each value of the objective layers. Default is 1.
		active_layers – Supply a list of active layers. If this parameter is supplied all 'learnabile' of all layers will be ignored and only 				these layers will be trained. By default, all the learnable layers are used.
		verbose – Similar to the rest of the toolbox.


   """

                    
    learning_rates = (0.05, 0.01)  

    net.train( epochs = (20), 
               k = 2,  
               pre_train_discriminator = 3,
               validate_after_epochs = 1,
               visualize_after_epochs = 1,
               training_accuracy = True,
               show_progress = True,
               early_terminate = True,
               verbose = verbose)

  """
    Training function of the network. Calling this will begin training.

		Parameters:	
			epochs – (num_epochs for each learning rate... ) to train Default is (20, 20)
			validate_after_epochs – 1, after how many epochs do you want to validate ?
			save_after_epochs – 1, Save network after that many epochs of training.
			show_progress – default is True, will display a clean progressbar. If verbose is 3 or more - False
			early_terminate – True will allow early termination.
			learning_rates – (annealing_rate, learning_rates ... ) length must be one more than epochs Default is (0.05, 0.01, 0.001)


  """
                           
    return net





if __name__ == '__main__':
    
    from yann.special.datasets import cook_mnist_normalized_zero_mean as c 
    import sys

    dataset = None  
    if len(sys.argv) > 1:
        if sys.argv[1] == 'create_dataset':
            data = c (verbose = 2)
            dataset = data.dataset_location()
        else:
            dataset = sys.argv[1]
    else:
        print "provide dataset"
    
    if dataset is None:
        print " creating a new dataset to run through"
        data = c (verbose = 2)
        dataset = data.dataset_location() 

    net = shallow_gan_mnist ( dataset, verbose = 2 )


