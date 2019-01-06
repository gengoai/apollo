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

import com.gengoai.apollo.linear.Axis;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.ModelParameters;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.vectorizer.NoOptVectorizer;
import com.gengoai.conversion.Cast;
import com.gengoai.io.Resources;
import com.gengoai.io.resource.Resource;
import com.gengoai.tuple.Tuple2;
import com.github.jcrfsuite.CrfTagger;
import com.github.jcrfsuite.util.Pair;
import third_party.org.chokkan.crfsuite.*;

import java.io.IOException;
import java.io.Serializable;
import java.util.Base64;
import java.util.List;

import static com.gengoai.Validation.notNull;
import static com.gengoai.tuple.Tuples.$;

/**
 * <p>Conditional Random Field sequence labeler wrapping CRFSuite.</p>
 *
 * @author David B. Bracewell
 */
public class Crf extends SequenceLabeler implements Serializable {
   private static final long serialVersionUID = 1L;
   private String modelFile;
   private volatile CrfTagger tagger;

   public Crf(Preprocessor... preprocessors) {
      this(new PreprocessorList(preprocessors));
   }

   public Crf(PreprocessorList preprocessors) {
      super(ModelParameters.create(new NoOptVectorizer<>())
                           .update(p -> {
                              p.preprocessors(preprocessors);
                              p.sequenceValidator = Validator.ALWAYS_TRUE;
                              p.featureVectorizer = new NoOptVectorizer<>();
                           }));
   }

   private Tuple2<ItemSequence, StringList> toItemSequence(Example sequence) {
      ItemSequence seq = new ItemSequence();
      StringList labels = new StringList();
      for (Example instance : sequence) {
         Item item = new Item();
         for (Feature feature : instance.getFeatures()) {
            item.add(new Attribute(feature.name, feature.value));
         }
         if (instance.hasLabel()) {
            labels.add(instance.getLabelAsString());
         }
         seq.add(item);
      }
      return $(seq, labels);
   }

   @Override
   protected SequenceLabeler fitPreprocessed(Dataset dataset, FitParameters parameters) {
      Parameters fitParameters = Cast.as(notNull(parameters), Parameters.class);
      CrfSuiteLoader.INSTANCE.load();
      Trainer trainer = new Trainer();
      dataset.forEach(sequence -> {
         Tuple2<ItemSequence, StringList> instance = toItemSequence(sequence);
         trainer.append(instance.v1, instance.v2, 0);
      });
      trainer.select(fitParameters.crfSolver.parameterSetting, "crf1d");
      trainer.set("max_iterations", Integer.toString(fitParameters.maxIterations));
      trainer.set("c2", Double.toString(fitParameters.c2));
      trainer.set("c1", Double.toString(fitParameters.c1));
      trainer.set("epsilon", Double.toString(fitParameters.eps));
      trainer.set("feature.minfreq", Integer.toString(fitParameters.minFeatureFreq));
      modelFile = Resources.temporaryFile().asFile().get().getAbsolutePath();
      trainer.train(modelFile, -1);
      trainer.clear();
      tagger = new CrfTagger(modelFile);
      return this;
   }

   @Override
   public Labeling label(Example sequence) {
      CrfSuiteLoader.INSTANCE.load();
      List<Pair<String, Double>> tags = tagger.tag(toItemSequence(preprocess(sequence)).v1);
      return new Labeling(tags.stream().map(Pair::getFirst).toArray(String[]::new));
   }


   @Override
   public Parameters getDefaultFitParameters() {
      return new Crf.Parameters();
   }

   @Override
   public int getNumberOfLabels() {
      return tagger.getlabels().size();
   }

   private void readObject(java.io.ObjectInputStream stream) throws IOException, ClassNotFoundException {
      CrfSuiteLoader.INSTANCE.load();
      Resource tmp = Resources.temporaryFile();
      int length = stream.readInt();
      byte[] bytes = new byte[length];
      stream.readFully(bytes);
      tmp.write(Base64.getDecoder().decode(bytes));
      this.modelFile = tmp.asFile().get().getAbsolutePath();
      this.tagger = new CrfTagger(modelFile);
   }

   private Tuple2<ItemSequence, StringList> toItemSequence(NDArray sequence, boolean includeLabel) {
      ItemSequence seq = new ItemSequence();
      StringList labels = new StringList();
      NDArray y = null;
      if (includeLabel) {
         y = sequence.getLabelAsNDArray();
         if (y.isMatrix()) {
            y = y.argMax(Axis.ROW);
         }
      }
      for (int row = 0; row < sequence.numRows(); row++) {
         Item item = new Item();
         sequence.sparseRowIterator(row)
                 .forEachRemaining(e -> item.add(new Attribute(
                    Integer.toString(e.getColumn()),
                    e.getValue()
                 )));
         if (includeLabel) {
            labels.add(Integer.toString((int) y.get(row)));
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
   public static class Parameters extends FitParameters {
      private static final long serialVersionUID = 1L;
      /**
       * The coefficient for L2 regularization (default 1.0)
       */
      public double c2 = 1.0;
      /**
       * The coefficient for L1 regularization (default 0.0 - not used)
       */
      public double c1 = 0;
      /**
       * The type of solver to use (defaults to LBFGS)
       */
      public CrfSolver crfSolver = CrfSolver.LBFGS;
      /**
       * The maximum number of iterations (defaults to 200)
       */
      public int maxIterations = 200;
      /**
       * The minimumn number of times a feature must appear to be kept (default 0 - keep all)
       */
      public int minFeatureFreq = 0;
      /**
       * The epsilon parameter to determine convergence (default is 1e-5)
       */
      public double eps = 1e-5;


   }

}//END OF Crf
