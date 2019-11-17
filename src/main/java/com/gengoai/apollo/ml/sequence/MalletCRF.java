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

package com.gengoai.apollo.ml.sequence;

import cc.mallet.fst.*;
import cc.mallet.optimize.LimitedMemoryBFGS;
import cc.mallet.optimize.Optimizable;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.TokenSequence2FeatureVectorSequence;
import cc.mallet.types.*;
import cc.mallet.util.MalletLogger;
import com.gengoai.ParameterDef;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.Params;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.vectorizer.NoOptVectorizer;
import com.gengoai.conversion.Cast;

import java.util.Arrays;
import java.util.logging.Level;

/**
 * @author David B. Bracewell
 */
public class MalletCRF extends SequenceLabeler {
   public static final ParameterDef<Integer> THREADS = ParameterDef.intParam("numThreads");
   public static final ParameterDef<Order> ORDER = new ParameterDef<>("order", Order.class);
   private SerialPipes pipes;
   private CRF model;

   /**
    * Instantiates a new GreedyAvgPerceptron.
    *
    * @param preprocessors the preprocessors
    */
   public MalletCRF(Preprocessor... preprocessors) {
      this(new PreprocessorList(preprocessors));
   }

   /**
    * Instantiates a new GreedyAvgPerceptron.
    *
    * @param validator     the validator
    * @param preprocessors the preprocessors
    */
   public MalletCRF(SequenceValidator validator, Preprocessor... preprocessors) {
      this(validator, new PreprocessorList(preprocessors));
   }

   /**
    * Instantiates a new GreedyAvgPerceptron.
    *
    * @param preprocessors the preprocessors
    */
   public MalletCRF(PreprocessorList preprocessors) {
      this(SequenceValidator.ALWAYS_TRUE, preprocessors);
   }

   /**
    * Instantiates a new GreedyAvgPerceptron.
    *
    * @param validator     the validator
    * @param preprocessors the preprocessors
    */
   public MalletCRF(SequenceValidator validator, PreprocessorList preprocessors) {
      super(SequencePipeline.create(NoOptVectorizer.INSTANCE)
                            .update(p -> {
                               p.preprocessorList.addAll(preprocessors);
                               p.featureVectorizer = NoOptVectorizer.INSTANCE;
                               p.sequenceValidator = validator;
                            }));

   }

   @Override
   protected void fitPreprocessed(Dataset preprocessed, FitParameters fitParameters) {
      Parameters params = Cast.as(fitParameters);


      if( params.verbose.value()) {
         MalletLogger.getLogger(ThreadedOptimizable.class.getName())
                     .setLevel(Level.INFO);
         MalletLogger.getLogger(CRFTrainerByValueGradients.class.getName())
                     .setLevel(Level.INFO);
         MalletLogger.getLogger(CRF.class.getName())
                     .setLevel(Level.INFO);
         MalletLogger.getLogger(CRFOptimizableByBatchLabelLikelihood.class.getName())
                     .setLevel(Level.INFO);
         MalletLogger.getLogger(LimitedMemoryBFGS.class.getName())
                     .setLevel(Level.INFO);
      } else {
         MalletLogger.getLogger(ThreadedOptimizable.class.getName())
                     .setLevel(Level.OFF);
         MalletLogger.getLogger(CRFTrainerByValueGradients.class.getName())
                     .setLevel(Level.OFF);
         MalletLogger.getLogger(CRF.class.getName())
                     .setLevel(Level.OFF);
         MalletLogger.getLogger(CRFOptimizableByBatchLabelLikelihood.class.getName())
                     .setLevel(Level.OFF);
         MalletLogger.getLogger(LimitedMemoryBFGS.class.getName())
                     .setLevel(Level.OFF);

      }

      Alphabet dataAlphabet = new Alphabet();
      pipes = new SerialPipes(Arrays.asList(new SequenceToTokenSequence(),
                                            new TokenSequence2FeatureVectorSequence(dataAlphabet, false, true)));
      pipes.setDataAlphabet(dataAlphabet);
      pipes.setTargetAlphabet(new LabelAlphabet());

      InstanceList trainingData = new InstanceList(pipes);
      preprocessed.forEach(i -> {
         Label[] target = new Label[i.size()];
         LabelAlphabet labelAlphabet = Cast.as(trainingData.getTargetAlphabet());
         for (int i1 = 0; i1 < target.length; i1++) {
            target[i1] = labelAlphabet.lookupLabel(i.getExample(i1).getLabel(), true);
         }
         trainingData.addThruPipe(new Instance(i, new LabelSequence(target), null, null));
      });


      model = new CRF(pipes, null);
      switch (params.order.value()) {
         case ZERO:
            break;
         case FIRST:
            model.addFullyConnectedStatesForLabels();
            break;
         case SECOND:
            model.addFullyConnectedStatesForBiLabels();
            break;
         case THIRD:
            model.addFullyConnectedStatesForTriLabels();
            break;
      }

      model.setWeightsDimensionAsIn(trainingData, false);

      CRFOptimizableByBatchLabelLikelihood batchOptLabel = new CRFOptimizableByBatchLabelLikelihood(model,
                                                                                                    trainingData,
                                                                                                    params.numberOfThreads
                                                                                                       .value());
      ThreadedOptimizable optLabel = new ThreadedOptimizable(batchOptLabel,
                                                             trainingData,
                                                             model.getParameters().getNumFactors(),
                                                             new CRFCacheStaleIndicator(model));
      Optimizable.ByGradientValue[] opts = {optLabel};
      CRFTrainerByValueGradients crfTrainer = new CRFTrainerByValueGradients(model, opts);
      crfTrainer.setMaxResets(0);



      crfTrainer.train(trainingData, params.maxIterations.value());
      optLabel.shutdown();
   }

   public static class Parameters extends FitParameters<Parameters> {
      private static final long serialVersionUID = 1L;
      public final Parameter<Integer> numberOfThreads = parameter(THREADS, 8);
      public final Parameter<Order> order = parameter(ORDER, Order.FIRST);
      public final Parameter<Integer> maxIterations = parameter(Params.Optimizable.maxIterations, Integer.MAX_VALUE);
   }

   @Override
   public FitParameters<?> getFitParameters() {
      return new Parameters();
   }

   @Override
   public Labeling label(Example example) {
      Instance crfOutput = model.label(new Instance(example, null, null, null));
      ArraySequence<String> bestOutput = Cast.as(crfOutput.getTarget());
      Labeling labeling = new Labeling(example.size());
      for (int i = 0; i < example.size(); i++) {
         labeling.labels[i] = bestOutput.get(i);
      }
      return labeling;
   }
}//END OF MalletCRF
