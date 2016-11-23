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

package com.davidbracewell.apollo.ml.preprocess.filter;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.preprocess.InstancePreprocessor;
import com.davidbracewell.io.structured.ArrayValue;
import com.davidbracewell.io.structured.ElementType;
import com.davidbracewell.io.structured.StructuredReader;
import com.davidbracewell.io.structured.StructuredWriter;
import com.davidbracewell.string.StringUtils;
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
public class NameFilter implements FilterProcessor<Instance>, InstancePreprocessor, ArrayValue {
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
   public void fit(Dataset<Instance> dataset) {
   }

   @Override
   public void reset() {
   }

   @Override
   public String describe() {
      return "NameFilter{patterns=" + patterns + "}";
   }


   @Override
   public Instance apply(Instance example) {
      return Instance.create(
         example.getFeatures().stream().filter(f -> {
            for (Pattern pattern : patterns) {
               if (pattern.matcher(f.getName()).find()) {
                  return false;
               }
            }
            return true;
         }).collect(Collectors.toList()),
         example.getLabel()
                            );
   }

   @Override
   public void write(StructuredWriter writer) throws IOException {
      for (Pattern pattern : patterns) {
         writer.beginObject();
         writer.writeKeyValue("pattern", pattern.toString());
         writer.writeKeyValue("flags", pattern.flags());
         writer.endObject();
      }
   }

   @Override
   public boolean requiresFit() {
      return false;
   }

   @Override
   public void read(StructuredReader reader) throws IOException {
      reset();
      while (reader.peek() != ElementType.END_ARRAY) {
         reader.beginObject();
         int flags = -1;
         String pattern = StringUtils.EMPTY;
         while (reader.peek() != ElementType.END_OBJECT) {
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
   public String toString() {
      return describe();
   }

}//END OF NameFilter
