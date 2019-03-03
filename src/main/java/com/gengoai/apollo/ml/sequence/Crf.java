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

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.params.DoubleParam;
import com.gengoai.apollo.ml.params.IntParam;
import com.gengoai.apollo.ml.params.Param;
import com.gengoai.apollo.ml.params.ParamMap;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.apollo.ml.vectorizer.NoOptVectorizer;
import com.gengoai.io.Resources;
import com.gengoai.io.resource.Resource;
import com.gengoai.tuple.Tuple2;
import com.github.jcrfsuite.CrfTagger;
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
   public static final Param<CrfSolver> crfSolver = new Param<>("crfSolver", CrfSolver.class,
                                                                "The type of solver to use.");
   public static final DoubleParam C1 = new DoubleParam("c1", "The coefficient for L1 regularization ", i -> i >= 0);
   public static final DoubleParam C2 = new DoubleParam("c2", "The coefficient for L2 regularization ", i -> i >= 0);
   public static final IntParam minFeatureFreq = new IntParam("minFeatureFreq",
                                                              "The minimumn number of times a feature must appear to be kept",
                                                              i -> i >= 0);
   private static final long serialVersionUID = 1L;
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
   protected void fitPreprocessed(Dataset dataset, ParamMap fitParameters) {
      CrfSuiteLoader.INSTANCE.load();
      Trainer trainer = new Trainer();
      dataset.forEach(sequence -> {
         Tuple2<ItemSequence, StringList> instance = toItemSequence(sequence);
         trainer.append(instance.v1, instance.v2, 0);
      });
      trainer.select(fitParameters.get(crfSolver).parameterSetting, "crf1d");
      trainer.set("max_iterations", Integer.toString(fitParameters.get(maxIterations)));
      trainer.set("c2", Double.toString(fitParameters.get(C2)));
      trainer.set("c1", Double.toString(fitParameters.get(C1)));
      trainer.set("epsilon", Double.toString(fitParameters.get(tolerance)));
      trainer.set("feature.minfreq", Integer.toString(fitParameters.get(minFeatureFreq)));
      modelFile = Resources.temporaryFile().asFile().orElseThrow(IllegalArgumentException::new).getAbsolutePath();
      trainer.train(modelFile, -1);
      trainer.clear();
      tagger = new CrfTagger(modelFile);
   }

   @Override
   public ParamMap getFitParameters() {
      return new ParamMap(
         verbose.set(false),
         maxIterations.set(100),
         tolerance.set(1e-4),
         minFeatureFreq.set(0),
         C1.set(0.0),
         C2.set(1.0),
         crfSolver.set(CrfSolver.LBFGS),
         tolerance.set(1e-4)
      );
   }

   @Override
   public int getNumberOfLabels() {
      return tagger.getlabels().size();
   }

   @Override
   public Labeling label(Example sequence) {
      CrfSuiteLoader.INSTANCE.load();
      return new Labeling(tagger.tag(toItemSequence(getPipeline().preprocessorList.apply(sequence)).v1));
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
      for (Example instance : sequence) {
         Item item = new Item();
         for (Feature feature : instance.getFeatures()) {
            item.add(new Attribute(feature.getName(), feature.getValue()));
         }
         if (instance.hasLabel()) {
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


}//END OF Crf
