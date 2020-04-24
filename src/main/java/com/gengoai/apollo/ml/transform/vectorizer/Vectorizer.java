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

package com.gengoai.apollo.ml.transform.vectorizer;

import com.gengoai.Validation;
import com.gengoai.apollo.math.linalg.NDArray;
import com.gengoai.apollo.ml.DataSet;
import com.gengoai.apollo.ml.Datum;
import com.gengoai.apollo.ml.encoder.Encoder;
import com.gengoai.apollo.ml.observation.Observation;
import com.gengoai.apollo.ml.transform.Transform;
import lombok.Getter;
import lombok.NonNull;

import java.util.*;

/**
 * <p>
 * Base class for specialized {@link Transform}s that converts {@link Observation}s into {@link NDArray} observations.
 * Each vectorizer has an associated {@link Encoder} that learns the mapping of variable names into NDArray indexes.
 * </p>
 *
 * @author David B. Bracewell
 */
public abstract class Vectorizer implements Transform {
   private static final long serialVersionUID = 1L;
   @Getter
   protected final Encoder encoder;
   protected final Set<String> sources = new HashSet<>();

   protected Vectorizer(@NonNull Encoder encoder) {
      this.encoder = encoder;
   }

   @Override
   public final DataSet fitAndTransform(DataSet dataset) {
      if(!encoder.isFixed()) {
         encoder.fit(dataset.stream().flatMap(d -> d.stream(sources)));
      }
      return this.transform(dataset);
   }

   @Override
   public Set<String> getInputs() {
      return Collections.unmodifiableSet(sources);
   }

   @Override
   public Set<String> getOutputs() {
      return Collections.unmodifiableSet(sources);
   }

   public Transform sources(@NonNull String... sources) {
      return sources(Arrays.asList(sources));
   }

   public Transform sources(@NonNull Collection<String> sources) {
      Validation.checkArgument(sources.size() > 0, "No sources specified");
      this.sources.clear();
      this.sources.addAll(sources);
      return this;
   }

   @Override
   public DataSet transform(@NonNull DataSet dataset) {
      dataset = dataset.map(this::transform);
      for(String output : sources) {
         dataset.updateMetadata(output, m -> {
            m.setDimension(encoder.size());
            m.setType(NDArray.class);
            m.setEncoder(encoder);
         });
      }
      return dataset;
   }

   @Override
   public final Datum transform(@NonNull Datum datum) {
      for(String output : sources) {
         datum.update(output, this::transform);
      }
      return datum;
   }

   protected abstract NDArray transform(Observation observation);

}//END OF Vectorizer
