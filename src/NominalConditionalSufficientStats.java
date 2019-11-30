/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    NominalConditionalSufficientStats.java
 *    Copyright (C) 2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.trees.ht;

import java.io.Serializable;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;


import weka.core.Utils;

/**
 * Maintains sufficient stats for the distribution of a nominal attribute
 * 
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @author Mark Hall (mhall{[at]}pentaho{[dot]}com)
 * @version $Revision$
 */
public class NominalConditionalSufficientStats extends
  ConditionalSufficientStats implements Serializable {

  /**
   * For serialization
   */
  private static final long serialVersionUID = -669902060601313488L;

  /**
   * Inner class that implements a discrete distribution
   * 
   * @author Mark Hall (mhall{[at]}pentaho{[dot]}com)
   * 
   */
  protected class ValueDistribution implements Serializable {

    /**
     * For serialization
     */
    private static final long serialVersionUID = -61711544350888154L;

    protected final Map<Integer, WeightMass> m_dist = new LinkedHashMap<Integer, WeightMass>();

    private double m_sum;

    public void add(int val, double weight) {
      WeightMass count = m_dist.get(val);
      if (count == null) {
        count = new WeightMass();
        count.m_weight = 1.0;
        m_sum += 1.0;
        m_dist.put(val, count);
      }
      count.m_weight += weight;
      m_sum += weight;
    }

    public void delete(int val, double weight) {
      WeightMass count = m_dist.get(val);
      if (count != null) {
        count.m_weight -= weight;
        m_sum -= weight;
      }
    }

    public double getWeight(int val) {
      WeightMass count = m_dist.get(val);
      if (count != null) {
        return count.m_weight;
      }

      return 0.0;
    }

    public double sum() {
      return m_sum;
    }
  }
  
  protected class AttValueDiscriminationDist implements Serializable{
	 
	  private static final long serialVersionUID = 1L;
	   /**m_attValDiscDist: four discrimination stats of this specific value of 
	                        this nominal attribute and their weightmasses, i.e., counts*/
		protected final Map<String, WeightMass> m_attValDiscDist= new HashMap<String, WeightMass>(){
			{
				put(deprivedTotalDist, null);
				put(undeprivedTotalDist, null);
				put(deprivedGrantedDist, null);
				put(undeprivedGrantedDist, null);
			}
		}; 
		
		public void add(int senAttVal, String classVal, double weight) {
			if (senAttVal== undeprivedIndex) {
				update_m_attValDiscDist(undeprivedTotalDist, weight);
				if (classVal.equals(grantedString)) {
					update_m_attValDiscDist(undeprivedGrantedDist, weight);
				}
			}
			
			if (senAttVal== deprivedIndex) {
				update_m_attValDiscDist(deprivedTotalDist, weight);
				if (classVal.equals(grantedString)) {
					update_m_attValDiscDist(deprivedGrantedDist, weight);
				}
			}
		}
		
		public void update_m_attValDiscDist(String whichToUpdate, double weight) {
			WeightMass count= m_attValDiscDist.get(whichToUpdate);
			if (count == null) {
				count= new WeightMass();
				count.m_weight= 1.0;
				m_attValDiscDist.put(whichToUpdate, count);
			}
			count.m_weight+= weight;
		}
		
		public Map<String, WeightMass> get_m_attDist(){
			return m_attValDiscDist;
		}
		
		// calculate discrimination of this specific value of this nominal attribute
		public double calPerAttValDiscrimination() {
			double undeprivedTotalCount= 0, undeprivedGrantedCount= 0, deprivedGrantedCount= 0, deprivedTotalCount= 0;
			double undeprivedRate = 0;
			double deprivedRate = 0;
			
			if (m_attValDiscDist.get(deprivedTotalDist)!= null) {
				deprivedTotalCount = m_attValDiscDist.get(deprivedTotalDist).m_weight;
			}
			if (m_attValDiscDist.get(undeprivedTotalDist)!= null) {
				undeprivedTotalCount = m_attValDiscDist.get(undeprivedTotalDist).m_weight;
			}
			if (m_attValDiscDist.get(deprivedGrantedDist)!= null) {
				deprivedGrantedCount = m_attValDiscDist.get(deprivedGrantedDist).m_weight;
			}
			if (m_attValDiscDist.get(undeprivedGrantedDist)!= null) {
				undeprivedGrantedCount = m_attValDiscDist.get(undeprivedGrantedDist).m_weight;
			}
	

			if (undeprivedTotalCount != 0) {
				undeprivedRate = undeprivedGrantedCount / undeprivedTotalCount;
			}

			if (deprivedTotalCount != 0) {
				deprivedRate = deprivedGrantedCount / deprivedTotalCount;
			}

			return Math.abs(undeprivedRate- deprivedRate);
		}

	
	}
  
  protected double m_totalWeight;
  protected double m_missingWeight;

  @Override
  															/**added senAttVal*/
  public void update(double attVal, String classVal, double weight, int senAttVal) {
    if (Utils.isMissingValue(attVal)) {
      m_missingWeight += weight;
    } else {
      new Integer((int) attVal);
      ValueDistribution valDist = (ValueDistribution) m_classLookup
        .get(classVal);
      if (valDist == null) {
        valDist = new ValueDistribution();
        valDist.add((int) attVal, weight);
        m_classLookup.put(classVal, valDist);
      } else {
        valDist.add((int) attVal, weight);
      }
      
      /**update discrimination distribution of this nominal attribute*/
      AttValueDiscriminationDist attValDist= (AttValueDiscriminationDist) nomlAttValLookup.get((int) attVal);
      if (attValDist== null) {
    	  attValDist= new AttValueDiscriminationDist();
    	  attValDist.add(senAttVal, classVal, weight);
    	  nomlAttValLookup.put((int) attVal, attValDist);
      } else {
    	  attValDist.add(senAttVal, classVal, weight);
      }
    }

    m_totalWeight += weight;
  }
  
   /**
    * compute the discrimination of this nominal attribute
    * Pre-discrimination of this node: when the attribute is set as the sensitive attribute
    * Post discrimination: if this attribute is used for splitting
    */
	public double calNomlAttDiscrimination() {
		double discrimination= 0;
		double perAttValDiscrimination= 0;
		
		for (Map.Entry<Integer, Object> e: nomlAttValLookup.entrySet()) {
			perAttValDiscrimination= ((AttValueDiscriminationDist) e.getValue()).calPerAttValDiscrimination();
			discrimination+= perAttValDiscrimination;
		}
		
		return discrimination;
	}
  

  @Override
  public double probabilityOfAttValConditionedOnClass(double attVal,
    String classVal) {
    ValueDistribution valDist = (ValueDistribution) m_classLookup.get(classVal);
    if (valDist != null) {
      double prob = valDist.getWeight((int) attVal) / valDist.sum();
      return prob;
    }

    return 0;
  }

  protected List<Map<String, WeightMass>> classDistsAfterSplit() {

    // att index keys to class distribution
    Map<Integer, Map<String, WeightMass>> splitDists = new HashMap<Integer, Map<String, WeightMass>>();

    for (Map.Entry<String, Object> cls : m_classLookup.entrySet()) {
      String classVal = cls.getKey();
      ValueDistribution attDist = (ValueDistribution) cls.getValue();

      for (Map.Entry<Integer, WeightMass> att : attDist.m_dist.entrySet()) {
        Integer attVal = att.getKey();
        WeightMass attCount = att.getValue();

        Map<String, WeightMass> clsDist = splitDists.get(attVal);
        if (clsDist == null) {
          clsDist = new HashMap<String, WeightMass>();
          splitDists.put(attVal, clsDist);
        }

        WeightMass clsCount = clsDist.get(classVal);

        if (clsCount == null) {
          clsCount = new WeightMass();
          clsDist.put(classVal, clsCount);
        }

        clsCount.m_weight += attCount.m_weight;
      }

    }

    List<Map<String, WeightMass>> result = new LinkedList<Map<String, WeightMass>>();
    for (Map.Entry<Integer, Map<String, WeightMass>> v : splitDists.entrySet()) {
      result.add(v.getValue());
    }

    return result;
  }

  @Override
  /**Regarding nominal attribute, there is just one possibility for bestSplit*/
  public SplitCandidate bestSplit(SplitMetric splitMetric,
    Map<String, WeightMass> preSplitDist, String attName, ActiveHNode node) {

    List<Map<String, WeightMass>> postSplitDists = classDistsAfterSplit();
    /**
     * 1. merit is also being used when deciding the difference between first and second best;
     * 2. Pre and post discrimination are being calculated in InfoGainSplitMetricpreDiscrimination
     * 3. Pre discriminationthe is same for numeric attribute and nominal attribute splitting
     */
    
    double postDiscMerit= 0;
    
    ConditionalSufficientStats splitAttStats= node.get_m_nodeStats().get(attName);
    postDiscMerit= ((NominalConditionalSufficientStats)splitAttStats).calNomlAttDiscrimination();
    
    double merit = splitMetric.evaluateSplit(preSplitDist, postSplitDists, attName, postDiscMerit, node);
    SplitCandidate candidate = new SplitCandidate(
      new UnivariateNominalMultiwaySplit(attName), postSplitDists, merit);

    return candidate;
  }
}
