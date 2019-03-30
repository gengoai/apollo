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

import com.gengoai.Param;
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
   public static final Param<Boolean> verbose = Param.boolParam("verbose");

   /**
    * Common parameters for machine learning algorithms that use the Optimization package
    */
   public static final class Optimizable {
      /**
       * Maximum number of iterations to optimize for
       */
      public static final Param<Integer> maxIterations = Param.intParam("maxIterations");
      /**
       * The number iterations to report progress
       */
      public static final Param<Integer> reportInterval = Param.intParam("reportInterval");
      /**
       * The number of iterations to record metrics to determine convergence.
       */
      public static final Param<Integer> historySize = Param.intParam("historySize");
      /**
       * The number of examples to use in the batch for mini-batch optimization
       */
      public static final Param<Integer> batchSize = Param.intParam("batchSize");
      /**
       * The tolerance in change of metric from last iteration to this iteration for determining convergence.
       */
      public static final Param<Double> tolerance = Param.doubleParam("tolerance");
      /**
       * The learning rate
       */
      public static final Param<Double> learningRate = Param.doubleParam("learningRate");
      /**
       * The activation function here.
       */
      public static final Param<Activation> activation = new Param<>("activation", Activation.class);
      /**
       * The loss function to use
       */
      public static final Param<LossFunction> lossFunction = new Param<>("lossFunction", LossFunction.class);
      /**
       * True - cache data
       */
      public static final Param<Boolean> cacheData = Param.boolParam("cacheData");
      /**
       * Weight update methodology
       */
      public static final Param<WeightUpdate> weightUpdate = new Param<>("weightUpdate", WeightUpdate.class);
   }

   /**
    * The type Tree.
    */
   public static final class Tree {
      /**
       * The constant depthLimited.
       */
      public static final Param<Boolean> depthLimited = Param.boolParam("depthLimited");
      /**
       * The constant prune.
       */
      public static final Param<Boolean> prune = Param.boolParam("prune");
      /**
       * The constant maxDepth.
       */
      public static final Param<Integer> maxDepth = Param.intParam("maxDepth");
      /**
       * The constant minInstances.
       */
      public static final Param<Integer> minInstances = Param.intParam("minInstances");
   }

   /**
    * The type Clustering.
    */
   public static final class Clustering {
      /**
       * The constant K.
       */
      public static final Param<Integer> K = Param.intParam("K");
      /**
       * The constant measure.
       */
      public static final Param<Measure> measure = new Param<>("measure", Measure.class);
      /**
       * The constant linkage.
       */
      public static final Param<Linkage> linkage = new Param<>("linkage", Linkage.class);
   }

   /**
    * The type Embedding.
    */
   public static final class Embedding {
      /**
       * The constant dimension.
       */
      public static final Param<Integer> dimension = Param.intParam("dimension");
      /**
       * The constant windowSize.
       */
      public static final Param<Integer> windowSize = Param.intParam("windowSize");
   }


}//END OF Params
