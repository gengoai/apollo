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

package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Instance;
import com.google.common.base.Preconditions;
import com.google.common.collect.MinMaxPriorityQueue;
import com.google.common.collect.Ordering;
import lombok.Getter;
import lombok.NonNull;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.List;

/**
 * <p>Implementation of a Beam search decoder.</p>
 *
 * @author David B. Bracewell
 */
public class BeamDecoder implements Decoder, Serializable {
   private static final long serialVersionUID = 1L;
   @Getter
   private int beamSize;

   /**
    * Instantiates a new Beam decoder with beam size of 3.
    */
   public BeamDecoder() {
      this(3);
   }

   /**
    * Instantiates a new Beam decoder with the given beam size.
    *
    * @param beamSize the beam size
    */
   public BeamDecoder(int beamSize) {
      Preconditions.checkArgument(beamSize > 0, "Beam size must be > 0.");
      this.beamSize = beamSize;
   }

   @Override
   public Labeling decode(@NonNull SequenceLabeler model, @NonNull Sequence sequence) {
      if (sequence.size() == 0) {
         return new Labeling(0);
      }
      MinMaxPriorityQueue<DecoderState> queue = MinMaxPriorityQueue
                                                   .orderedBy(Ordering.natural().reverse())
                                                   .maximumSize(beamSize)
                                                   .create();
      queue.add(new DecoderState(0, null));
      List<DecoderState> newStates = new LinkedList<>();
      Context<Instance> iterator = sequence.iterator();
      while (iterator.hasNext()) {
         iterator.next();
         newStates.clear();
         while (!queue.isEmpty()) {
            DecoderState state = queue.remove();
            double[] result = model.estimate(
               iterator.getCurrent().getFeatures().iterator(),
               model.getTransitionFeatures().extract(state)
                                            );
            for (int i = 0; i < result.length; i++) {
               String label = model.getLabelEncoder().decode(i).toString();
               if (model.getValidator().isValid(label, state.tag, iterator.getCurrent())) {
                  newStates.add(new DecoderState(state, result[i], label));
               }
            }
            if (newStates.isEmpty()) {
               for (int i = 0; i < result.length; i++) {
                  String label = model.getLabelEncoder().decode(i).toString();
                  newStates.add(new DecoderState(state, result[i], label));
               }
            }
         }
         queue.addAll(newStates);
      }

      Labeling result = new Labeling(sequence.size());
      DecoderState last = queue.remove();
      while (last != null && last.tag != null) {
         result.setLabel(last.index - 1, last.tag, last.stateProbability);
         last = last.previousState;
      }
      queue.clear();
      return result;
   }

   /**
    * Sets the beam size.
    *
    * @param beamSize the beam size
    */
   public void setBeamSize(int beamSize) {
      Preconditions.checkArgument(beamSize > 0, "Beam size must be > 0.");
      this.beamSize = beamSize;
   }

}// END OF BeamDecoder
