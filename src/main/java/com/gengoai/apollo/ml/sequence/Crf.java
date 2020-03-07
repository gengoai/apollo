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

import com.gengoai.ParameterDef;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.Params;
import com.gengoai.apollo.ml.data.ExampleDataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.vectorizer.NoOptVectorizer;
import com.gengoai.conversion.Cast;
import com.gengoai.io.Resources;
import com.gengoai.io.resource.Resource;
import com.gengoai.jcrfsuite.CrfTagger;
import com.gengoai.tuple.Tuple2;
import third_party.org.chokkan.crfsuite.*;

import java.io.IOException;
import java.io.Serializable;
import java.util.Base64;

import static com.gengoai.tuple.Tuples.$;

/**
 * <p>Conditional Random Field sequence labeler wrapping CRFSuite.</p>
 *
 * @author David B. Bracewell
 */
public class Crf extends SequenceLabeler implements Serializable {
   private static final long serialVersionUID = 1L;
   public static final ParameterDef<Double> C2 = ParameterDef.doubleParam("c2");
   public static final ParameterDef<Double> C1 = ParameterDef.doubleParam("c1");
   public static final ParameterDef<Double> EPS = ParameterDef.doubleParam("eps");
   public static final ParameterDef<Integer> MIN_FEATURE_FREQ = ParameterDef.intParam("minFeatureFreq");
   public static final ParameterDef<CrfSolver> SOLVER = ParameterDef.param("solver", CrfSolver.class);
   private String modelFile;
   private volatile CrfTagger tagger;

   /**
    * Instantiates a new Crf with the given preprocessors.
    *
    * @param preprocessors the preprocessors
    */
   public Crf(Preprocessor... preprocessors) {
      this(new PreprocessorList(preprocessors));
   }

   /**
    * Instantiates a new Crf with the given preprocessors.
    *
    * @param preprocessors the preprocessors
    */
   public Crf(PreprocessorList preprocessors) {
      super(SequencePipeline.create(NoOptVectorizer.INSTANCE)
                            .update(p -> {
                               p.preprocessorList.addAll(preprocessors);
                               p.sequenceValidator = SequenceValidator.ALWAYS_TRUE;
                               p.featureVectorizer = NoOptVectorizer.INSTANCE;
                            }));
   }

   @Override
   protected void fitPreprocessed(ExampleDataset dataset, FitParameters parameters) {
      Parameters fitParameters = Cast.as(parameters);
      CrfSuiteLoader.INSTANCE.load();
      Trainer trainer = new Trainer();
      dataset.forEach(sequence -> {
         Tuple2<ItemSequence, StringList> instance = toItemSequence(sequence);
         trainer.append(instance.v1, instance.v2, 0);
      });
      trainer.select(fitParameters.crfSolver.value().parameterSetting, "crf1d");
      trainer.set("max_iterations", Integer.toString(fitParameters.maxIterations.value()));
      trainer.set("c2", Double.toString(fitParameters.c2.value()));
      trainer.set("c1", Double.toString(fitParameters.c1.value()));
      trainer.set("epsilon", Double.toString(fitParameters.eps.value()));
      trainer.set("feature.minfreq", Integer.toString(fitParameters.minFeatureFreq.value()));
      modelFile = Resources.temporaryFile().asFile().orElseThrow(IllegalArgumentException::new).getAbsolutePath();
      trainer.train(modelFile, -1);
      trainer.clear();
      tagger = new CrfTagger(modelFile);
   }

   @Override
   public Parameters getFitParameters() {
      return new Crf.Parameters();
   }

   @Override
   public int getNumberOfLabels() {
      return tagger.getlabels().size();
   }

   @Override
   public Labeling label(Example sequence) {
      CrfSuiteLoader.INSTANCE.load();
      ItemSequence itemSequence = toItemSequence(getPipeline().preprocessorList.apply(sequence)).v1;
      Labeling labeling = new Labeling(tagger.tag(itemSequence));
      deleteItemSequence(itemSequence);
      return labeling;
   }

   private void deleteItemSequence(ItemSequence itemSequence) {
      for(int i = 0; i < itemSequence.size(); i++) {
         itemSequence.get(i).delete();
      }
      itemSequence.delete();
   }

   private void readObject(java.io.ObjectInputStream stream) throws Exception {
      CrfSuiteLoader.INSTANCE.load();
      Resource tmp = Resources.temporaryFile();
      int length = stream.readInt();
      byte[] bytes = new byte[length];
      stream.readFully(bytes);
      tmp.write(Base64.getDecoder().decode(bytes));
      this.modelFile = tmp.asFile().orElseThrow(IllegalArgumentException::new).getAbsolutePath();
      this.tagger = new CrfTagger(modelFile);
   }

   private Tuple2<ItemSequence, StringList> toItemSequence(Example sequence) {
      ItemSequence seq = new ItemSequence();
      StringList labels = new StringList();
      for(Example instance : sequence) {
         Item item = new Item();
         for(Feature feature : instance.getFeatures()) {
            item.add(new Attribute(feature.getName(), feature.getValue()));
         }
         if(instance.hasLabel()) {
            labels.add(instance.getDiscreteLabel());
         }
         seq.add(item);
      }
      return $(seq, labels);
   }

   private void writeObject(java.io.ObjectOutputStream stream) throws IOException {
      byte[] modelBytes = Base64.getEncoder().encode(Resources.from(modelFile).readBytes());
      stream.writeInt(modelBytes.length);
      stream.write(modelBytes);
   }

   /**
    * Specialized Fit Parameters for use with CRFSuite.
    */
   public static class Parameters extends FitParameters<Parameters> {
      private static final long serialVersionUID = 1L;
      /**
       * The maximum number of iterations (defaults to 200)
       */
      public final Parameter<Integer> maxIterations = parameter(Params.Optimizable.maxIterations, 250);
      /**
       * The type of solver to use (defaults to LBFGS)
       */
      public final Parameter<CrfSolver> crfSolver = parameter(SOLVER, CrfSolver.LBFGS);
      /**
       * The coefficient for L1 regularization (default 0.0 - not used)
       */
      public final Parameter<Double> c1 = parameter(C1, 0d);
      /**
       * The coefficient for L2 regularization (default 1.0)
       */
      public final Parameter<Double> c2 = parameter(C2, 1d);
      /**
       * The epsilon parameter to determine convergence (default is 1e-5)
       */
      public final Parameter<Double> eps = parameter(EPS, 1e-5);
      /**
       * The minimumn number of times a feature must appear to be kept (default 0 - keep all)
       */
      public final Parameter<Integer> minFeatureFreq = parameter(MIN_FEATURE_FREQ, 0);


   }

}//END OF Crf
