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
 */

package com.davidbracewell.apollo.ml.sequence.linear;

import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.LabelEncoder;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.apollo.ml.sequence.*;
import com.davidbracewell.io.Resources;
import com.davidbracewell.io.resource.Resource;
import com.github.jcrfsuite.CrfTagger;
import com.github.jcrfsuite.util.Pair;
import lombok.NonNull;
import third_party.org.chokkan.crfsuite.Attribute;
import third_party.org.chokkan.crfsuite.Item;
import third_party.org.chokkan.crfsuite.ItemSequence;

import java.io.IOException;
import java.util.Base64;
import java.util.Iterator;
import java.util.List;

/**
 * <p>Sequence Labeler wrapping CRFSuite.</p>
 *
 * @author David B. Bracewell
 */
public class CRFTagger extends SequenceLabeler {
   private static final long serialVersionUID = 1L;
   private volatile String modelFile;
   private transient volatile CrfTagger tagger;

   /**
    * Instantiates a new Model.
    *
    * @param labelEncoder       the label encoder
    * @param featureEncoder     the feature encoder
    * @param preprocessors      the preprocessors
    * @param transitionFeatures the transition features
    * @param modelFile          the model file
    * @param validator          the validator
    */
   public CRFTagger(@NonNull LabelEncoder labelEncoder, @NonNull Encoder featureEncoder, @NonNull PreprocessorList<Sequence> preprocessors, @NonNull TransitionFeatures transitionFeatures, String modelFile, @NonNull SequenceValidator validator) {
      super(labelEncoder, featureEncoder, preprocessors, transitionFeatures, validator);
      this.modelFile = modelFile;
      this.tagger = new CrfTagger(modelFile);
   }

   @Override
   public Labeling label(@NonNull Sequence sequence) {
      LibraryLoader.INSTANCE.load();
      ItemSequence seq = new ItemSequence();
      for (Instance instance : sequence.asInstances()) {
         Item item = new Item();
         for (Feature f : instance) {
            item.add(new Attribute(f.getName(), f.getValue()));
         }
         seq.add(item);
      }
      List<Pair<String, Double>> tags = tagger.tag(seq);
      Labeling lr = new Labeling(sequence.size());
      for (int i = 0; i < tags.size(); i++) {
         lr.setLabel(i, tags.get(i).first, tags.get(i).second);
      }
      return lr;
   }

   @Override
   public double[] estimate(Iterator<Feature> observation, Iterator<String> transitions) {
      return null;
   }

   private void writeObject(java.io.ObjectOutputStream stream) throws IOException {
      byte[] modelBytes = Base64.getEncoder().encode(Resources.from(modelFile).readBytes());
      stream.writeInt(modelBytes.length);
      stream.write(modelBytes);
   }

   private void readObject(java.io.ObjectInputStream stream) throws IOException, ClassNotFoundException {
      LibraryLoader.INSTANCE.load();
      Resource tmp = Resources.temporaryFile();
      int length = stream.readInt();
      byte[] bytes = new byte[length];
      stream.readFully(bytes);
      tmp.write(Base64.getDecoder().decode(bytes));
      this.modelFile = tmp.asFile().get().getAbsolutePath();
      this.tagger = new CrfTagger(modelFile);
   }


}// END OF CRFTagger
