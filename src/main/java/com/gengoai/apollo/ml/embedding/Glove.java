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

package com.gengoai.apollo.ml.embedding;

import com.gengoai.Param;
import com.gengoai.Stopwatch;
import com.gengoai.apollo.linear.DenseMatrix;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.DiscretePipeline;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.Params;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.collection.counter.Counter;
import com.gengoai.collection.counter.Counters;
import com.gengoai.conversion.Cast;
import com.gengoai.logging.Logger;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * <p>Implementation of Glove.</p>
 *
 * @author David B. Bracewell
 */
public class Glove extends Embedding {
   private static final Logger log = Logger.getLogger(Glove.class);
   private static final long serialVersionUID = 1L;

   /**
    * Instantiates a new Glove.
    *
    * @param preprocessors the preprocessors
    */
   public Glove(Preprocessor... preprocessors) {
      super(preprocessors);
   }

   /**
    * Instantiates a new Glove.
    *
    * @param modelParameters the model parameters
    */
   public Glove(DiscretePipeline modelParameters) {
      super(modelParameters);
   }

   @Override
   protected void fitPreprocessed(Dataset preprocessed, FitParameters fitParameters) {
      Parameters p = Cast.as(fitParameters);
      Stopwatch sw = Stopwatch.createStarted();

      double size = preprocessed.size();
      double processed = 0;
      Counter<IntIntPair> counts = Counters.newCounter();
      for (Example example : preprocessed) {
         List<Integer> ids = toIndices(example);
         for (int i = 1; i < ids.size(); i++) {
            int iW = ids.get(i);
            for (int j = Math.max(0, i - p.windowSize.value()); j < i; j++) {
               int jW = ids.get(j);
               double incrementBy = 1.0 / (i - j);
               counts.increment(new IntIntPair(iW, jW), incrementBy);
               counts.increment(new IntIntPair(jW, iW), incrementBy);
            }
         }
         processed++;
         if (processed % 1000 == 0) {
            System.out.println("processed " + (100 * processed / size));
         }
      }

      sw.stop();
      if (p.verbose.value()) {
         log.info("Cooccurrence Matrix computed in {0}", sw);
      }

      List<Cooccurrence> cooccurrences = new ArrayList<>();
      counts.forEach((e, v) -> {
         if (getPipeline().featureVectorizer.getString(e.i).equals("the")) {
            System.out.println(getPipeline().featureVectorizer.getString(e.j) + "\t" + v);
         }
         cooccurrences.add(new Cooccurrence(e.i, e.j, v));
      });
      counts.clear();


      DoubleMatrix[] W = new DoubleMatrix[getNumberOfFeatures() * 2];
      DoubleMatrix[] gradSq = new DoubleMatrix[getNumberOfFeatures() * 2];
      for (int i = 0; i < getNumberOfFeatures() * 2; i++) {
         W[i] = DoubleMatrix.rand(p.dimension.value()).sub(0.5f).divi(p.dimension.value());
         gradSq[i] = DoubleMatrix.ones(p.dimension.value());
      }

      DoubleMatrix biases = DoubleMatrix.rand(getNumberOfFeatures() * 2).sub(0.5f).divi(p.dimension.value());
      DoubleMatrix gradSqBiases = DoubleMatrix.ones(getNumberOfFeatures() * 2);


      int vocabLength = getNumberOfFeatures();

      for (int itr = 0; itr < p.maxIterations.value(); itr++) {
         double globalCost = 0d;
         Collections.shuffle(cooccurrences);

         for (Cooccurrence cooccurrence : cooccurrences) {
            int iWord = cooccurrence.word1;
            int iContext = cooccurrence.word2 + vocabLength;
            double count = cooccurrence.count;

            DoubleMatrix v_main = W[iWord];
            double b_main = biases.get(iWord);
            DoubleMatrix gradsq_W_main = gradSq[iWord];
            double gradsq_b_main = gradSqBiases.get(iWord);

            DoubleMatrix v_context = W[iContext];
            double b_context = biases.get(iContext);
            DoubleMatrix gradsq_W_context = gradSq[iContext];
            double gradsq_b_contenxt = gradSqBiases.get(iContext);


            double diff = v_main.dot(v_context) + b_main + b_context - Math.log(count);
            double fdiff = count > p.xMax.value()
                           ? diff
                           : Math.pow(count / p.xMax.value(), p.alpha.value()) * diff;

            globalCost += 0.5 * fdiff * diff;

            fdiff *= p.learningRate.value();
            //Gradients for word vector terms
            DoubleMatrix grad_main = v_context.mmul(fdiff);
            DoubleMatrix grad_context = v_main.mmul(fdiff);

            v_main.subi(grad_main.divi(MatrixFunctions.sqrt(gradsq_W_main)));
            v_context.subi(grad_context.divi(MatrixFunctions.sqrt(gradsq_W_context)));
            gradsq_W_main.addi(MatrixFunctions.pow(grad_context, 2));
            gradsq_W_context.addi(MatrixFunctions.pow(grad_main, 2));

            biases.put(iWord, b_main - fdiff / Math.sqrt(gradsq_b_main));
            biases.put(iContext, b_context - fdiff / Math.sqrt(gradsq_b_contenxt));
            fdiff *= fdiff;

            gradSqBiases.put(iWord, gradSqBiases.get(iWord) + fdiff);
            gradSqBiases.put(iContext, gradSqBiases.get(iContext) + fdiff);
         }

         if (p.verbose.value()) {
            log.info("Iteration: {0},  cost:{1}", (itr + 1), globalCost / cooccurrences.size());
         }

      }

      NDArray[] vectors = new NDArray[getNumberOfFeatures()];
      for (int i = 0; i < vocabLength; i++) {
         W[i].addi(W[i + vocabLength]);
         String k = getPipeline().featureVectorizer.getString(i);
         vectors[i] = new DenseMatrix(W[i]);
         vectors[i].setLabel(k);
      }
      this.vectorIndex = new DefaultVectorIndex(vectors);
   }

   @Override
   public Parameters getFitParameters() {
      return new Parameters();
   }

   private List<Integer> toIndices(Example sequence) {
      List<Integer> out = new ArrayList<>();
      for (Example example : sequence) {
         if (example.getFeatures().size() > 0) {
            out.add(getPipeline().featureVectorizer.indexOf(example.getFeatures().get(0).getName()));
         }
      }
      return out;
   }

   /**
    * The type Cooccurrence.
    */
   public static class Cooccurrence {
      /**
       * The Count.
       */
      public final double count;
      /**
       * The Word 1.
       */
      public final int word1;
      /**
       * The Word 2.
       */
      public final int word2;

      /**
       * Instantiates a new Cooccurrence.
       *
       * @param word1 the word 1
       * @param word2 the word 2
       * @param count the count
       */
      public Cooccurrence(int word1, int word2, double count) {
         this.word1 = word1;
         this.word2 = word2;
         this.count = count;
      }
   }

   private static class IntIntPair implements Serializable {
      private static final long serialVersionUID = 1L;
      /**
       * The .
       */
      final int i;
      /**
       * The J.
       */
      final int j;

      private IntIntPair(int i, int j) {
         this.i = i;
         this.j = j;
      }

      @Override
      public boolean equals(Object obj) {
         if (this == obj) {return true;}
         if (obj == null || getClass() != obj.getClass()) {return false;}
         final IntIntPair other = (IntIntPair) obj;
         return Objects.equals(this.i, other.i)
                   && Objects.equals(this.j, other.j);
      }

      @Override
      public int hashCode() {
         return 31 * Integer.hashCode(i) + 37 * Integer.hashCode(j);
      }
   }

   public static final Param<Double> alpha = Param.doubleParam("alpha");
   public static final Param<Integer> xMax = Param.intParam("xMax");

   /**
    * The type Parameters.
    */
   public static class Parameters extends EmbeddingFitParameters<Parameters> {
      /**
       * The Alpha.
       */
      public final Parameter<Double> alpha = parameter(Glove.alpha, 0.75);
      /**
       * The Learning rate.
       */
      public final Parameter<Double> learningRate = parameter(Params.Optimizable.learningRate, 0.05);
      /**
       * The X max.
       */
      public final Parameter<Integer> xMax = parameter(Glove.xMax, 100);
      /**
       * The Max iterations.
       */
      public final Parameter<Integer> maxIterations = parameter(Params.Optimizable.maxIterations, 25);
   }
}//END OF Glove
