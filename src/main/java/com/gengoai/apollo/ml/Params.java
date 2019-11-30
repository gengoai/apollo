/*
 * (c) 2005 David B. Bracewell
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 */

package com.gengoai.apollo.ml;

import com.gengoai.ParameterDef;
import com.gengoai.apollo.ml.clustering.Linkage;
import com.gengoai.apollo.optimization.WeightUpdate;
import com.gengoai.apollo.optimization.activation.Activation;
import com.gengoai.apollo.optimization.loss.LossFunction;
import com.gengoai.apollo.statistics.measure.Measure;

/**
 * Common parameters for machine learning algorithms
 *
 * @author David B. Bracewell
 */
public final class Params {

   /**
    * True - Verbose output during training
    */
   public static final ParameterDef<Boolean> verbose = ParameterDef.boolParam("verbose");

   /**
    * Common parameters for machine learning algorithms that use the Optimization package
    */
   public static final class Optimizable {
      /**
       * Maximum number of iterations to optimize for
       */
      public static final ParameterDef<Integer> maxIterations = ParameterDef.intParam("maxIterations");
      /**
       * The number iterations to report progress
       */
      public static final ParameterDef<Integer> reportInterval = ParameterDef.intParam("reportInterval");
      /**
       * The number of iterations to record metrics to determine convergence.
       */
      public static final ParameterDef<Integer> historySize = ParameterDef.intParam("historySize");
      /**
       * The number of examples to use in the batch for mini-batch optimization
       */
      public static final ParameterDef<Integer> batchSize = ParameterDef.intParam("batchSize");
      /**
       * The tolerance in change of metric from last iteration to this iteration for determining convergence.
       */
      public static final ParameterDef<Double> tolerance = ParameterDef.doubleParam("tolerance");
      /**
       * The learning rate
       */
      public static final ParameterDef<Double> learningRate = ParameterDef.doubleParam("learningRate");
      /**
       * The activation function here.
       */
      public static final ParameterDef<Activation> activation = ParameterDef.param("activation", Activation.class);
      /**
       * The loss function to use
       */
      public static final ParameterDef<LossFunction> lossFunction = ParameterDef.param("lossFunction", LossFunction.class);
      /**
       * True - cache data
       */
      public static final ParameterDef<Boolean> cacheData = ParameterDef.boolParam("cacheData");
      /**
       * Weight update methodology
       */
      public static final ParameterDef<WeightUpdate> weightUpdate = ParameterDef.param("weightUpdate", WeightUpdate.class);
   }

   /**
    * The type Tree.
    */
   public static final class Tree {
      /**
       * The constant depthLimited.
       */
      public static final ParameterDef<Boolean> depthLimited = ParameterDef.boolParam("depthLimited");
      /**
       * The constant prune.
       */
      public static final ParameterDef<Boolean> prune = ParameterDef.boolParam("prune");
      /**
       * The constant maxDepth.
       */
      public static final ParameterDef<Integer> maxDepth = ParameterDef.intParam("maxDepth");
      /**
       * The constant minInstances.
       */
      public static final ParameterDef<Integer> minInstances = ParameterDef.intParam("minInstances");
   }

   /**
    * The type Clustering.
    */
   public static final class Clustering {
      /**
       * The constant K.
       */
      public static final ParameterDef<Integer> K = ParameterDef.intParam("K");
      /**
       * The constant measure.
       */
      public static final ParameterDef<Measure> measure = ParameterDef.param("measure", Measure.class);
      /**
       * The constant linkage.
       */
      public static final ParameterDef<Linkage> linkage = ParameterDef.param("linkage", Linkage.class);
   }

   /**
    * The type Embedding.
    */
   public static final class Embedding {
      /**
       * The constant dimension.
       */
      public static final ParameterDef<Integer> dimension = ParameterDef.intParam("dimension");
      /**
       * The constant windowSize.
       */
      public static final ParameterDef<Integer> windowSize = ParameterDef.intParam("windowSize");
   }


}//END OF Params
