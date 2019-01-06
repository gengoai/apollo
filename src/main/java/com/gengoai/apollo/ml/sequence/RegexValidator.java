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

import java.io.Serializable;
import java.util.regex.Pattern;

/**
 * @author David B. Bracewell
 */
public class RegexValidator implements Validator, Serializable {
   public static final Validator BASIC_IOB_VALIDATOR = new RegexValidator(true, "(O|__BOS__)", "I-*");
   private static final long serialVersionUID = 1L;
   private final Pattern pattern;
   private final boolean inverted;

   public RegexValidator(String pattern) {
      this(false, pattern);
   }


   public RegexValidator(boolean inverted, String... patterns) {
      StringBuilder regex = new StringBuilder();
      for (int i = 0; i < patterns.length; i++) {
         if (i > 0) {
            regex.append('\0');
         }
         regex.append(patterns[i].replace("*", ".*"));
      }
      this.pattern = Pattern.compile(regex.toString());
      this.inverted = inverted;
   }


   @Override
   public boolean isValid(String currentLabel, String previousLabel, Example instance) {
      previousLabel = (previousLabel == null) ? "__BOS__" : previousLabel;
      String lbl = previousLabel + "\0" + currentLabel;
      boolean found = pattern.matcher(lbl).matches();
      if (inverted) {
         return !found;
      }
      return found;
   }
}//END OF RegexValidator
