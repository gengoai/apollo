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
import com.davidbracewell.apollo.ml.preprocess.InstancePreprocessor;

import java.io.Serializable;
import java.util.HashSet;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * The type Name filter.
 *
 * @author David B. Bracewell
 */
public class NameFilter implements FilterProcessor<Instance>, InstancePreprocessor, Serializable {
  private static final long serialVersionUID = 1L;
  private final Set<Pattern> patterns = new HashSet<>();

  /**
   * Instantiates a new Name filter.
   *
   * @param patterns the patterns
   */
  public NameFilter(String... patterns) {
    for (String pattern : patterns) {
      this.patterns.add(Pattern.compile(pattern));
    }
  }

  @Override
  public void visit(Instance example) {

  }

  @Override
  public Instance process(Instance example) {
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
  public void finish() {

  }

  @Override
  public void reset() {

  }

  @Override
  public String describe() {
    return "NameFilter: " + patterns;
  }

}//END OF RemoveFilter
