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

package com.gengoai.apollo.ml;

import com.gengoai.string.Strings;

import java.io.Serializable;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * <P>A feature extractor that allows for the extraction of contextual features, i.e. using features from previous and
 * subsequent examples. Context features are defined using the feature prefix and a relative offset to the current
 * example as follows:</P>
 * <code>
 * PREFIX[relative_offset]
 * </code>
 * <p>
 * where relative offset is an integer with a negative representing to the left, optional positive sign representing to
 * the right, and 0 representing the current example. A more concrete example is:
 * </p>
 * <code>
 * WORD[-1]|POS[-1]|WORD[0]|WORD[+1]
 * </code>
 * <p>
 * in which we declare a feature made up of the previous <code>WORD</code> and <code>POS</code>, the current
 * <code>WORD</code>, and the next <code>WORD</code>. As illustrated in the example multiple prefixes can be specified
 * for a feature using a pipe <code>|</code> to separate. The contextual patterns generate boolean features in the
 * following form:
 * </p>
 * <code>
 * PREFIX1[relative_offset]|PREFIX2[relative_offset]=SUFFIX1|SUFFIX2
 * </code>
 * <p>
 * Contextual features are generated by iterating over the examples in a sequence using a {@link ContextualIterator},
 * which will return dummy beginning of sequence and end of sequence features when needed. These can be ignored and no
 * features generated if the context does not exist, by prepending a tilde <code>~</code> to the pattern, e.g.
 * <code>~WORD[-1]|WORD[0]</code>, will only create features if <code>WORD[-1]</code> exists.
 * </p>
 * <p>
 * ContextFeaturizers are required to be used in conjunction with {@link Featurizer}s when converting input data into
 * {@link Example}s. This is accomplished by creating a {@link Featurizer} and then calling {@link
 * Featurizer#withContext(String[])}.
 * </p>
 *
 * @param <I> the input type parameter
 * @author David B. Bracewell
 */
public abstract class ContextFeaturizer<I> implements FeatureExtractor<I>, Serializable {
   private static final long serialVersionUID = 1L;

   /**
    * Chains together multiple <code>ContextFeaturizer</code>s into a single Featurizer.
    *
    * @param <I>      the input type parameter
    * @param patterns the patterns
    * @return the context featurizer
    */
   public static <I> ContextFeaturizer<I> chain(String... patterns) {
      return new ChainedContextFeaturizer<>(Stream.of(patterns)
                                                  .map(SingleContextFeaturizer::new)
                                                  .collect(Collectors.toList()));
   }

   /**
    * Convenience method for creating a ContextFeaturizer from a given pattern. Will use beginning and end of sequence
    * markers as needed.
    *
    * @param <I>     the input type parameter
    * @param pattern the pattern
    * @return the context feature
    */
   public static <I> ContextFeaturizer<I> contextFeaturizer(String pattern) {
      return new SingleContextFeaturizer<>(pattern);
   }

   @Override
   public final Example extract(I input) {
      throw new UnsupportedOperationException();
   }

   private static class ChainedContextFeaturizer<I> extends ContextFeaturizer<I> {
      private static final long serialVersionUID = 1L;
      private final List<ContextFeaturizer<? super I>> contextFeaturizers;

      private ChainedContextFeaturizer(List<ContextFeaturizer<? super I>> contextFeaturizers) {
         this.contextFeaturizers = contextFeaturizers;
      }

      @Override
      public Example contextualize(Example sequence) {
         for (ContextFeaturizer<? super I> contextFeaturizer : contextFeaturizers) {
            sequence = contextFeaturizer.contextualize(sequence);
         }
         return sequence;
      }
   }

   private static class SingleContextFeaturizer<I> extends ContextFeaturizer<I> {
      private static final Pattern featurePattern = Pattern.compile("^(.+?)\\[([-+]?\\d+)\\]$");
      private static final long serialVersionUID = 1L;
      private final String featurePrefix;
      private final boolean ignoreEmptyContext;
      private final int[] offsets;
      private final String[] prefix;


      private SingleContextFeaturizer(String pattern) {
         this.ignoreEmptyContext = pattern.startsWith("~");
         String[] patterns = this.ignoreEmptyContext
                             ? pattern.substring(1).split("\\|")
                             : pattern.split("\\|");
         prefix = new String[patterns.length];
         offsets = new int[patterns.length];
         featurePrefix = Strings.join(patterns, "|");
         for (int i = 0; i < patterns.length; i++) {
            Matcher m = featurePattern.matcher(patterns[i]);
            if (m.find()) {
               prefix[i] = m.group(1) + "=";
               offsets[i] = Integer.parseInt(m.group(2));
            } else {
               throw new IllegalArgumentException(patterns[i] + " is not a legal pattern");
            }
         }
      }

      @Override
      public Example contextualize(Example sequence) {
         for (ContextualIterator itr = sequence.iterator(); itr.hasNext(); ) {
            Example instance = itr.next();

            StringBuilder fName = new StringBuilder(featurePrefix).append("=");
            for (int i = 0; i < prefix.length; i++) {
               Feature f = itr.getContext(offsets[i]).getFeatureByPrefix(prefix[i]);
               if (ignoreEmptyContext) {
                  if (f.getSuffix().startsWith("__BOS-") || f.getSuffix().startsWith("__EOS-")) {
                     fName = null;
                     break;
                  }
               }
               if (i > 0) {
                  fName.append("|");
               }
               fName.append(f.getSuffix());
            }
            if (fName != null) {
               instance.getFeatures().add(Feature.booleanFeature(fName.toString()));
            }
         }
         return sequence;
      }
   }


}//END OF ContextFeaturizer
