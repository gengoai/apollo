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
import com.gengoai.apollo.ml.data.ExampleDataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.vectorizer.NoOptVectorizer;
import com.gengoai.conversion.Cast;

import java.util.Arrays;
import java.util.logging.Level;
import java.util.regex.Pattern;

import static com.gengoai.collection.Arrays2.arrayOfInt;

/**
 * @author David B. Bracewell
 */
public class MalletCRF extends SequenceLabeler {
   private static final long serialVersionUID = 1L;
   public static final ParameterDef<Integer> THREADS = ParameterDef.intParam("numThreads");
   public static final ParameterDef<Order> ORDER = ParameterDef.param("order", Order.class);
   public static final ParameterDef<Boolean> FULLY_CONNECTED = ParameterDef.boolParam("fullyConnected");
   public static final ParameterDef<String> START_STATE = ParameterDef.strParam("startState");
   private SerialPipes pipes;
   private CRF model;
   private String startState;

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
   public MalletCRF(MalletSequenceValidator validator, Preprocessor... preprocessors) {
      this(validator, new PreprocessorList(preprocessors));
   }

   /**
    * Instantiates a new GreedyAvgPerceptron.
    *
    * @param preprocessors the preprocessors
    */
   public MalletCRF(PreprocessorList preprocessors) {
      this(null, preprocessors);
   }

   /**
    * Instantiates a new GreedyAvgPerceptron.
    *
    * @param validator     the validator
    * @param preprocessors the preprocessors
    */
   public MalletCRF(MalletSequenceValidator validator, PreprocessorList preprocessors) {
      super(SequencePipeline.create(NoOptVectorizer.INSTANCE)
                  .update(p -> {
                     p.preprocessorList.addAll(preprocessors);
                     p.featureVectorizer = NoOptVectorizer.INSTANCE;
                     p.sequenceValidator = validator;
                  }));

   }

   @Override
   protected void fitPreprocessed(ExampleDataset preprocessed, FitParameters fitParameters) {
      Parameters params = Cast.as(fitParameters);


      if(params.verbose.value()) {
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
      }
      else {
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
         for(int i1 = 0; i1 < target.length; i1++) {
            target[i1] = labelAlphabet.lookupLabel(i.getExample(i1).getLabel(), true);
         }
         trainingData.addThruPipe(new Instance(i, new LabelSequence(target), null, null));
      });


      model = new CRF(pipes, null);

      int[] order = {};
      switch(params.order.value()) {
         case FIRST:
            order = arrayOfInt(1);
            break;
         case SECOND:
            order = arrayOfInt(1, 2);
            break;
         case THIRD:
            order = arrayOfInt(1, 2, 3);
            break;
      }

      MalletSequenceValidator sv = Cast.as(getSequenceValidator());
      Pattern allowed = sv == null
                        ? null
                        : sv.getAllowed();
      Pattern forbidden = sv == null
                          ? null
                          : sv.getForbidden();
      model.addOrderNStates(trainingData,
                            order,
                            null,
                            params.startState.value(),
                            forbidden,
                            allowed,
                            params.fullyConnected.value());
      this.startState = params.startState.value();
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

   @Override
   public FitParameters<?> getFitParameters() {
      return new Parameters();
   }

   @Override
   public Labeling label(Example example) {
      Sequence<?> sequence = Cast.as(model.getInputPipe()
                                           .instanceFrom(new Instance(example, null, null, null)).getData());
      Sequence<?> bestOutput = model.transduce(sequence);
      SumLattice lattice = new SumLatticeDefault(model, sequence, true);
      Transducer.State sj = model.getState(startState);
      Labeling labeling = new Labeling(example.size());
      for(int i = 0; i < sequence.size(); i++) {
         Transducer.State si = model.getState((String) bestOutput.get(i));
         labeling.labels[i] = (String) bestOutput.get(i);
         double pS = lattice.getGammaProbability(i, si);
         double PSjSi = lattice.getXiProbability(i, sj, si);
         labeling.scores[i] = Math.max(pS, PSjSi);
         sj = si;
      }
      return labeling;
   }

   public static class Parameters extends FitParameters<Parameters> {
      private static final long serialVersionUID = 1L;
      public final Parameter<Integer> numberOfThreads = parameter(THREADS, 20);
      public final Parameter<Order> order = parameter(ORDER, Order.FIRST);
      public final Parameter<Integer> maxIterations = parameter(Params.Optimizable.maxIterations, 250);
      public final Parameter<Boolean> fullyConnected = parameter(FULLY_CONNECTED, true);
      public final Parameter<String> startState = parameter(START_STATE, "0");
   }
}//END OF MalletCRF
