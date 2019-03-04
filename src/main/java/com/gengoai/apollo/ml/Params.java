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
 * @author David B. Bracewell
 */
public final class Params {

   public static final Param<Boolean> verbose = Param.boolParam("verbose");

   public static final class Optimizable {
      public static final Param<Integer> maxIterations = Param.intParam("maxIterations");
      public static final Param<Integer> reportInterval = Param.intParam("reportInterval");
      public static final Param<Integer> historySize = Param.intParam("historySize");
      public static final Param<Integer> batchSize = Param.intParam("batchSize");
      public static final Param<Double> tolerance = Param.doubleParam("tolerance");
      public static final Param<Double> learningRate = Param.doubleParam("learningRate");
      public static final Param<Activation> activation = new Param<>("activation", Activation.class);
      public static final Param<LossFunction> lossFunction = new Param<>("lossFunction", LossFunction.class);
      public static final Param<Boolean> cacheData = Param.boolParam("cacheData");
      public static final Param<WeightUpdate> weightUpdate = new Param<>("weightUpdate", WeightUpdate.class);
   }

   public static final class Tree {
      public static final Param<Boolean> depthLimited = Param.boolParam("depthLimited");
      public static final Param<Boolean> prune = Param.boolParam("prune");
      public static final Param<Integer> maxDepth = Param.intParam("maxDepth");
      public static final Param<Integer> minInstances = Param.intParam("minInstances");
   }

   public static final class Clustering {
      public static final Param<Integer> K = Param.intParam("K");
      public static final Param<Measure> measure = new Param<>("measure", Measure.class);
      public static final Param<Linkage> linkage = new Param<>("linkage", Linkage.class);
   }

   public static final class Embedding {
      public static final Param<Integer> dimension = Param.intParam("dimension");
      public static final Param<Integer> windowSize = Param.intParam("windowSize");
   }


}//END OF Params
