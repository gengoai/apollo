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

package com.gengoai.apollo.ml.feature;

import com.gengoai.conversion.Cast;
import com.gengoai.parsing.Lexer;
import com.gengoai.parsing.ParserToken;
import com.gengoai.parsing.TokenDef;
import com.gengoai.parsing.TokenStream;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static com.gengoai.string.Re.*;

/**
 * @author David B. Bracewell
 */
public enum ContextPatternParser implements TokenDef {
   STRICT {
      @Override
      public List<List<FeatureGetter>> generate(ParserToken token) {
         return Collections.emptyList();
      }

      @Override
      public String getPattern() {
         return "~";
      }
   },
   PIPE {
      @Override
      public List<List<FeatureGetter>> generate(ParserToken token) {
         return Collections.emptyList();
      }

      @Override
      public String getPattern() {
         return "\\|";
      }
   },
   WINDOW {
      @Override
      public List<List<FeatureGetter>> generate(ParserToken token) {
         List<FeatureGetter> getters = new ArrayList<>();
         int low = Integer.parseInt(token.getVariable(0));
         int high = Integer.parseInt(token.getVariable(1));
         String[] prefixes = token.getVariable(2).split("\\s*,\\s*");
         for(int i = low; i <= high; i++) {
            final int offset = i;
            getters.addAll(Stream.of(prefixes)
                              .map(p -> new FeatureGetter(offset, p))
                              .collect(Collectors.toList()));
         }
         return List.of(getters);
      }

      @Override
      public String getPattern() {
         return re("WINDOW",
                   q("["),
                   namedGroup("", zeroOrOne(chars("-+")), oneOrMore(DIGIT)),
                   q(".."),
                   namedGroup("", zeroOrOne(chars("-+")), oneOrMore(DIGIT)),
                   q("]"),
                   q("("),
                   namedGroup("",
                              greedyOneOrMore(NON_WHITESPACE),
                              zeroOrOne("\\s*,\\s*"),
                              greedyOneOrMore(NON_WHITESPACE)),
                   q(")"));
      }
   },
   BIGRAM {
      @Override
      public String getPattern() {
         return re("BIGRAM",
                   q("["),
                   namedGroup("", zeroOrOne(chars("-+")), oneOrMore(DIGIT)),
                   q(".."),
                   namedGroup("", zeroOrOne(chars("-+")), oneOrMore(DIGIT)),
                   q("]"),
                   q("("),
                   namedGroup("",
                              greedyOneOrMore(NON_WHITESPACE),
                              zeroOrOne("\\s*,\\s*"),
                              greedyOneOrMore(NON_WHITESPACE)),
                   q(")"));
      }

      @Override
      public List<List<FeatureGetter>> generate(ParserToken token) {
         List<List<FeatureGetter>> getters = new ArrayList<>();
         int low = Integer.parseInt(token.getVariable(0));
         int high = Integer.parseInt(token.getVariable(1));
         String[] prefixes = token.getVariable(2).split("\\s*,\\s*");
         for(int i = low; i < high; i++) {
            final int offset = i;
            getters.add(Stream.of(prefixes)
                              .flatMap(p -> Stream.of(new FeatureGetter(offset, p),
                                                      new FeatureGetter(offset + 1, p)))
                              .collect(Collectors.toList()));
         }
         return getters;
      }
   },
   TRIGRAM {
      @Override
      public String getPattern() {
         return re("TRIGRAM",
                   q("["),
                   namedGroup("", zeroOrOne(chars("-+")), oneOrMore(DIGIT)),
                   q(".."),
                   namedGroup("", zeroOrOne(chars("-+")), oneOrMore(DIGIT)),
                   q("]"),
                   q("("),
                   namedGroup("", greedyOneOrMore(NON_WHITESPACE)),
                   q(")"));
      }

      @Override
      public List<List<FeatureGetter>> generate(ParserToken token) {
         List<List<FeatureGetter>> getters = new ArrayList<>();
         int low = Integer.parseInt(token.getVariable(0));
         int high = Integer.parseInt(token.getVariable(1));
         String[] prefixes = token.getVariable(2).split("\\s*,\\s*");
         for(int i = low; i < high - 1; i++) {
            final int offset = i;
            getters.add(Stream.of(prefixes)
                              .flatMap(p -> Stream.of(new FeatureGetter(offset, p),
                                                      new FeatureGetter(offset + 1, p),
                                                      new FeatureGetter(offset + 2, p)))
                              .collect(Collectors.toList()));
         }
         return getters;
      }
   },
   PREFIX {
      @Override
      public List<List<FeatureGetter>> generate(ParserToken token) {
         return List.of(List.of(new FeatureGetter(Integer.parseInt(token.getVariable(1)), token.getVariable(0))));
      }

      @Override
      public String getPattern() {
         return re(namedGroup("", greedyOneOrMore(any())),
                   q("["),
                   namedGroup("", zeroOrOne(chars("-+")), oneOrMore(DIGIT)),
                   q("]"));
      }
   };

   private static final Lexer lexer = Lexer.create(ContextPatternParser.values());

   public static <T> List<ContextFeaturizer<T>> parse(String pattern) {
      TokenStream ts = lexer.lex(pattern);
      List<List<FeatureGetter>> extractors = Collections.emptyList();

      AtomicBoolean isStrict = new AtomicBoolean(false);
      while(ts.hasNext()) {
         ParserToken token = ts.consume();
         if(token.getType() == STRICT) {
            isStrict.set(true);
            continue;
         }
         List<List<FeatureGetter>> getters = ((ContextPatternParser) token.getType()).generate(token);
         if(ts.hasNext()) {
            ts.consume(PIPE);
         }
         if(extractors.isEmpty()) {
            extractors = getters;
         } else {
            List<List<FeatureGetter>> out = new ArrayList<>();
            for(List<FeatureGetter> extractor : extractors) {
               for(List<FeatureGetter> getter : getters) {
                  out.add(new ArrayList<>(extractor));
                  out.get(out.size() - 1).addAll(getter);
               }
            }
            extractors = out;
         }
      }
      return Cast.cast(extractors.stream()
                                 .map(l -> new ContextFeaturizerImpl<>( isStrict.get(), l))
                                 .collect(Collectors.toList()));
   }

   protected abstract List<List<FeatureGetter>> generate(ParserToken token);

}
