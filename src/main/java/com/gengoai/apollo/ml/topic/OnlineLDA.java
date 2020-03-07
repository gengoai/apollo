/*
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
 */

package com.gengoai.apollo.ml.topic;

import com.gengoai.ParameterDef;
import com.gengoai.Validation;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.DiscretePipeline;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.Params;
import com.gengoai.apollo.ml.data.ExampleDataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.vectorizer.DiscreteVectorizer;
import com.gengoai.collection.Iterables;
import com.gengoai.collection.counter.Counter;
import com.gengoai.collection.counter.Counters;
import com.gengoai.conversion.Cast;
import lombok.NonNull;
import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.special.Gamma;

import java.util.List;
import java.util.function.Consumer;

public class OnlineLDA extends TopicModel {
   private static final long serialVersionUID = 1L;
   public static final ParameterDef<Double> alpha = ParameterDef.doubleParam("alpha");
   public static final ParameterDef<Double> eta = ParameterDef.doubleParam("eta");
   public static final ParameterDef<Double> kappa = ParameterDef.doubleParam("kappa");
   public static final ParameterDef<Double> tau0 = ParameterDef.doubleParam("tau0");

   public OnlineLDA(Preprocessor... preprocessors) {
      super(preprocessors);
   }

   public OnlineLDA(DiscretePipeline modelParameters) {
      super(modelParameters);
   }

   public static OnlineLDAFitParameters fitParameters() {
      return new OnlineLDAFitParameters();
   }

   public static OnlineLDAFitParameters fitParameters(@NonNull Consumer<OnlineLDAFitParameters> updater) {
      OnlineLDAFitParameters fp = fitParameters();
      updater.accept(fp);
      return fp;
   }

   @Override
   public double[] estimate(Example Example) {
      return new double[0];
   }

   @Override
   public NDArray getTopicDistribution(String feature) {
      return null;
   }

   private NDArray gammaSample(int r, int c) {
      final GammaDistribution dist = new GammaDistribution(100d, 0.01d);
      return NDArrayFactory.DENSE.array(r, c)
                                 .mapi(d -> dist.sample());
   }

   private NDArray dirichletExpectation(NDArray array) {
      NDArray vector = array.rowSums().mapi(Gamma::digamma);
      return array.map(Gamma::digamma)
                  .subiColumnVector(vector);
   }

   private void eStep(ModelP m, List<NDArray> batch) {
      m.gamma = gammaSample(batch.size(), m.K);
      var eLogTheta = dirichletExpectation(m.gamma);
      var expELogTheta = eLogTheta.map(Math::exp);
      m.stats = m.lambda.zeroLike();

      for(int i = 0; i < batch.size(); i++) {
         NDArray n = batch.get(i);
         int[] ids = n.sparseIndices();
         if(ids.length == 0) {
            continue;
         }
         NDArray nv = NDArrayFactory.DENSE.array(ids.length);
         for(int i1 = 0, i2 = 0; i1 < ids.length; i1++, i2++) {
            nv.set(i2, n.get(ids[i1]));
         }
         var gammaD = m.gamma.getRow(i);
         var expELogThetaD = expELogTheta.getRow(i);
         var expELogBetaD = m.expELogBeta.selectColumns(ids);
         var phiNorm = expELogThetaD.mmul(expELogBetaD).addi(1E-100);

         NDArray lastGamma;
         for(int iteration = 0; iteration < 100; iteration++) {
            lastGamma = gammaD;
            var v1 = nv.div(phiNorm).mmul(expELogBetaD.T());
            gammaD = expELogThetaD.mul(v1).addi(m.alpha);
            var eLogThetaD = dirichletExpectation(gammaD);
            expELogThetaD = eLogThetaD.map(Math::exp);
            phiNorm = expELogThetaD.mmul(expELogBetaD).addi(1E-100);
            if(gammaD.map(lastGamma, (d1, d2) -> Math.abs(d1 - d2)).mean() < 0.001) {
               break;
            }
         }
         m.gamma.setRow(i, gammaD);
         var o = outer(expELogThetaD, nv.div(phiNorm));
         for(int k = 0; k < ids.length; k++) {
            m.stats.incrementiColumn(ids[k], o.getColumn(k));
         }
      }
      m.stats.muli(m.expELogBeta);
   }

   @Override
   protected void fitPreprocessed(ExampleDataset preprocessed, FitParameters<?> fitParameters) {
      OnlineLDAFitParameters p = Validation.notNull(Cast.as(fitParameters, OnlineLDAFitParameters.class));
      final ModelP model = new ModelP();
      final int W = getPipeline().featureVectorizer.size();

      model.lambda = gammaSample(p.K.value(), W);
      model.eLogBeta = dirichletExpectation(model.lambda);
      model.expELogBeta = model.eLogBeta.map(Math::exp);
      model.stats = model.lambda.zeroLike();
      model.K = p.K.value();
      model.alpha = p.alpha.value();

      final double D = preprocessed.size();
      double tau0 = p.tau0.value();
      double kappa = p.kappa.value();
      double eta = p.eta.value();
      int batchSize = p.batchSize.value();
      int batchCount = 0;
      for(ExampleDataset docs : Iterables.asIterable(preprocessed.batchIterator(batchSize))) {
         List<NDArray> batch = docs.toVectorizedDataset(getPipeline()).stream().collect();
         eStep(model, batch);
         double rho = Math.pow(tau0 + batchCount, -kappa);
         NDArray a = model.lambda.mul(1 - rho);
         NDArray b = model.stats.mul(D / batch.size()).addi(eta);
         model.lambda = a.add(b);
         model.eLogBeta = dirichletExpectation(model.lambda);
         model.expELogBeta = model.eLogBeta.map(Math::exp);
         batchCount++;
      }
      model.lambda.diviColumnVector(model.lambda.rowSums());
      for(int i = 0; i < p.K.value(); i++) {
         NDArray topic = model.lambda.getRow(i);
         final DiscreteVectorizer dv = getPipeline().featureVectorizer;
         Counter<String> cntr = Counters.newCounter();
         topic.forEachSparse((fi, v) -> cntr.set(dv.getString(fi), v));
         for(String s : cntr.topN(10).itemsByCount(false)) {
            System.out.printf("%s (%.5f)  ", s, cntr.get(s));
         }
         System.out.println();
      }
   }

   private NDArray outer(NDArray vector, NDArray matrix) {
      NDArray out = NDArrayFactory.DENSE.array((int) vector.length(), (int) matrix.length());
      for(long i = 0; i < vector.length(); i++) {
         for(long j = 0; j < matrix.length(); j++) {
            out.set((int) i, (int) j, vector.get(i) * matrix.get(j));
         }
      }
      return out;
   }

   @Override
   public FitParameters<?> getFitParameters() {
      return new OnlineLDAFitParameters();
   }

   private static class ModelP {
      NDArray lambda;
      NDArray eLogBeta;
      NDArray expELogBeta;
      NDArray stats;
      NDArray gamma;
      int K;
      double alpha;
   }

   public static class OnlineLDAFitParameters extends com.gengoai.apollo.ml.FitParameters<OnlineLDAFitParameters> {
      public final Parameter<Integer> K = parameter(Params.Clustering.K, 100);
      public final Parameter<Integer> batchSize = parameter(Params.Optimizable.batchSize, 512);
      public final Parameter<Double> alpha = parameter(OnlineLDA.alpha, 0.1);
      public final Parameter<Double> eta = parameter(OnlineLDA.eta, 0.01);
      public final Parameter<Double> tau0 = parameter(OnlineLDA.tau0, 1.0);
      public final Parameter<Double> kappa = parameter(OnlineLDA.kappa, 0.75);
   }

}//END OF OnlineLDA
