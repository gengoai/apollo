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

package com.gengoai.apollo.ml.preprocess.filter;

import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.InstancePreprocessor;
import com.gengoai.json.JsonArraySerializable;
import com.gengoai.json.JsonReader;
import com.gengoai.json.JsonTokenType;
import com.gengoai.json.JsonWriter;
import com.gengoai.string.StringUtils;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.InstancePreprocessor;
import lombok.NonNull;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * <p>Removes all features whose name match any of the given regular expressions.</p>
 *
 * @author David B. Bracewell
 */
public class NameFilter implements FilterProcessor<Instance>, InstancePreprocessor, JsonArraySerializable {
   private static final long serialVersionUID = 1L;
   private final Set<Pattern> patterns = new HashSet<>();

   /**
    * Instantiates a new Name filter.
    *
    * @param patterns the patterns of feature names to remove
    */
   public NameFilter(@NonNull String... patterns) {
      for (String pattern : patterns) {
         this.patterns.add(Pattern.compile(pattern));
      }
   }

   /**
    * Instantiates a new Name filter.
    */
   protected NameFilter() {
   }

   @Override
   public Instance apply(Instance example) {
      return Instance.create(
         example.getFeatures().stream().filter(f -> {
            for (Pattern pattern : patterns) {
               if (pattern.matcher(f.getFeatureName()).find()) {
                  return false;
               }
            }
            return true;
         }).collect(Collectors.toList()),
         example.getLabel()
                            );
   }

   @Override
   public String describe() {
      return "NameFilter{patterns=" + patterns + "}";
   }

   @Override
   public void fit(Dataset<Instance> dataset) {
   }

   @Override
   public void fromJson(JsonReader reader) throws IOException {
      reset();
      while (reader.peek() != JsonTokenType.END_ARRAY) {
         reader.beginObject();
         int flags = -1;
         String pattern = StringUtils.EMPTY;
         while (reader.peek() != JsonTokenType.END_OBJECT) {
            switch (reader.peekName()) {
               case "pattern":
                  pattern = reader.nextKeyValue().v2.asString();
                  break;
               case "flags":
                  flags = reader.nextKeyValue().v2.asIntegerValue();
                  break;
            }
         }
         patterns.add(Pattern.compile(pattern, flags));
         reader.endObject();
      }
   }

   @Override
   public boolean requiresFit() {
      return false;
   }

   @Override
   public void reset() {
   }

   @Override
   public void toJson(JsonWriter writer) throws IOException {
      for (Pattern pattern : patterns) {
         writer.beginObject();
         writer.property("pattern", pattern.toString());
         writer.property("flags", pattern.flags());
         writer.endObject();
      }
   }

   @Override
   public String toString() {
      return describe();
   }

}//END OF NameFilter
