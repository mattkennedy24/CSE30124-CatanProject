import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Board } from '../models/board.model';
import { Tile } from '../models/tile.model';

@Component({
  selector: 'app-chat',
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.css']
})
export class ChatComponent implements OnInit {
  catanBoard: any;
  dataSet1: any;
  recs: any;
  first_recs: any;
  second_recs: any;

  positionChartUrl: string | null = null;
  diceChartUrl: string | null = null;
  featureChartUrl:  string | null = null;

  // Booleans to control visibility
  showFirstRecommendation = false;
  showSecondRecommendation = false;
  showPositionImportance = false;
  showBoardGraphic = false;
  showFeaturePlot = false;
  showDicePlot = false;


  constructor(
    private route: ActivatedRoute,
    private http: HttpClient
  ) {}

  ngOnInit(): void {
    this.route.queryParams.subscribe(params => {
      this.catanBoard = JSON.parse(params['catanBoard']);
      console.log('User Board data:', this.catanBoard);
    });
    this.sendBoardData(this.catanBoard)
  }
  // Define a method to send the user's catan board to the flask api for processing and to give user recommendation for settlement placement
  sendBoardData(boardData: any): void {
    const plainBoardData = JSON.parse(JSON.stringify(boardData));
    console.log(plainBoardData)
    const apiUrl = 'http://127.0.0.1:5000/api/recommend_settlement'; 
  
    this.http.post(apiUrl, { plainBoardData }).subscribe(
      (response: any) => {
        this.recs = response.recommendation;
        console.log('Recommendation Response:', response.recommendation);
        // Process the recommendation response here
      },
      (error) => {
        console.error('Error sending board data:', error);
      }
    );
  }


  // Method to handle the first settlement question
  askFirstSettlement() {
    this.first_recs = this.recs[0];
    this.showFirstRecommendation = true;
    this.showBoardGraphic = true;
  }

  // Method to handle the second settlement question
  askSecondSettlement() {
    this.second_recs = this.recs[1];
    this.showSecondRecommendation = true;
    this.showBoardGraphic = true;

  }

  // Add a method to get object keys
  getObjectKeys(obj: any): string[] {
    return Object.keys(obj);
  }

  askPosition() {
    // Add logic to fetch position importance chart
    const apiUrl = 'http://127.0.0.1:5000/api/position_importance_chart';
  
    this.http.get(apiUrl, { responseType: 'blob' }).subscribe(
      (response: Blob) => {
        const blob = new Blob([response], { type: 'image/png' });
        this.positionChartUrl = URL.createObjectURL(blob);
        this.showPositionImportance = true;
      },
      (error) => {
        console.error('Error fetching position importance chart:', error);
      }
    );
  }

  askDice() {
    // Add logic to fetch position importance chart
    const apiUrl = 'http://127.0.0.1:5000/api/dice_roll_plot';
  
    this.http.get(apiUrl, { responseType: 'blob' }).subscribe(
      (response: Blob) => {
        const blob = new Blob([response], { type: 'image/png' });
        this.diceChartUrl = URL.createObjectURL(blob);
        this.showDicePlot = true;
      },
      (error) => {
        console.error('Error fetching position importance chart:', error);
      }
    );
  }

  askFeatureImportance() {
    // Add logic to fetch position importance chart
    const apiUrl = 'http://127.0.0.1:5000/api/feature_importance_plot';
  
    this.http.get(apiUrl, { responseType: 'blob' }).subscribe(
      (response: Blob) => {
        const blob = new Blob([response], { type: 'image/png' });
        this.featureChartUrl = URL.createObjectURL(blob);
        this.showFeaturePlot = true;
      },
      (error) => {
        console.error('Error fetching position importance chart:', error);
      }
    );
  }

}

