import selenium
from selenium import webdriver as wd
from selenium.webdriver.common.by import By
import time
import pandas as pd
import os
from datetime import datetime
import subprocess

# 현재 날짜를 YYMMDD 형식으로 가져오기
current_date = datetime.now().strftime('%y%m%d')

# 폴더 생성
folder_name = f'키움_{current_date}'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# 1 Chrome 열기
driver = wd.Chrome()

teps = ['HitterBasic/Basic1', 'HitterBasic/Basic2', 'HitterBasic/Detail1', 'Defense/Basic', 'Runner/Basic']
Hitter_years = [43]
Defense_years = [24]
Runner_years = Defense_years
batting_order = range(2,14)
team = ['WO']

# 빈 결과 테이블
result_dfs = {year: [] for year in range(2019, 2019 + len(Hitter_years))}

# 열 이름 사전
column_translations = {
    # 타자기록
    '2B': '2루타', '3B': '3루타', 'AB': '타수', 'AO': '뜬공', 'AVG': '타율',
    'BB': '볼넷', 'BB/K': '볼넷/삼진', 'CS': '도루실패', 'E': '실책', 'G': '경기',
    'GDP': '병살타', 'GO': '땅볼', 'GO/AO': '땅볼/뜬공', 'GPA': '(1.8x출루율+장타율)/4',
    'GW RBI': '결승타', 'H': '안타', 'HBP': '사구', 'HR': '홈런', 'IBB': '고의4구',
    'ISOP': '순수장타율', 'MH': '멀티히트', 'OBP': '출루율', 'OPS': '출루율+장타율',
    'P/PA': '투구수/타석', 'PA': '타석', 'PH-BA': '대타타율', 'R': '득점', 'RBI': '타점',
    'RISP': '득점권타율', 'SAC': '희생번트', 'SB': '도루', 'SF': '희생플라이', 'SLG': '장타율',
    'SO': '삼진', 'TB': '루타', 'XBH': '장타', 'XR': '추정득점',
    # 수비기록
    'A': '어시스트', 'CS': '도루저지', 'CS%': '도루저지율', 'DP': '병살', 'FPCT': '수비율',
    'GS': '선발경기', 'PB': '포일', 'PKO': '견제사', 'PO': '풋아웃', 'POS': '포지션',
    'SB': '도루허용', 'IP' : '수비이닝',
    # 주루기록
    'OOB': '주루사', 'SBA': '도루시도', 'SB%': '도루성공률'
}

# 타순 매핑
order_mapping = {
    2: '1번', 3: '2번', 4: '3번', 5: '4번', 6: '5번',
    7: '6번', 8: '7번', 9: '8번', 10: '9번',
    11: '상위(1~2번)', 12: '중심(3~5번)', 13: '하위(6~9번)'
}

# 사이트 접속
for tep in teps[:2] :
    url = f'https://www.koreabaseball.com/Record/Player/{tep}.aspx'
    driver.get(url)
    time.sleep(3)

    # 타자 기록 접속
    for year_index, year in enumerate(Hitter_years):
        # 연도 선택
        Season_selector = f'#cphContents_cphContents_cphContents_ddlSeason_ddlSeason > option:nth-child({year})'
        driver.find_element(By.CSS_SELECTOR, Season_selector).click()
        time.sleep(3)
            
        # 팀 정보 선택 ('WO'로 설정)
        team_option = driver.find_element(By.CSS_SELECTOR, f"option[value='{team[0]}']")
        team_option.click()
        time.sleep(3)

        # 1페이지로 돌아가기
        first_page_button = driver.find_element(By.CSS_SELECTOR, f'#cphContents_cphContents_cphContents_ucPager_btnNo1')
        first_page_button.click()
        time.sleep(3)

        # 페이지 반복하여 데이터 가져오기 (1페이지와 2페이지)
        for page in range(1, 3):
            if page > 1:
                try:
                    next_page_button = driver.find_element(By.CSS_SELECTOR, f'#cphContents_cphContents_cphContents_ucPager_btnNo{page}')
                    next_page_button.click()
                    time.sleep(3)
                except:
                    break  # 페이지 버튼이 없는 경우 루프를 종료

            # 결과 테이블 가져오기
            result_table = driver.find_element(By.CSS_SELECTOR, '#cphContents_cphContents_cphContents_udpContent > div.record_result')
            table_html = result_table.getAttribute('outerHTML')

            # DataFrame으로 변환하여 리스트에 추가
            df = pd.read_html(table_html, encoding='utf-8')[0]

            # HitterBasic/Basic2일 경우에만 불필요한 열 제거
            if tep == 'HitterBasic/Basic2':
                df = df.drop(columns=['순위', '선수명', '팀명', 'AVG'])

            # 결과를 리스트에 추가
            result_dfs[2019 + year_index].append(df)

        # 연도가 바뀔 때마다 다시 첫 페이지로 돌아가기
        driver.get(url)
        time.sleep(3)
        driver.find_element(By.CSS_SELECTOR, Season_selector).click()
        time.sleep(3)

# 각 연도별로 DataFrame을 CSV 파일로 저장
for year, dfs in result_dfs.items():
    # dfs 개수에 따라 다르게 저장
    if len(dfs) >= 4:  # HitterBasic/Basic2의 데이터가 존재하는 경우
        df1 = dfs[0]  # HitterBasic/Basic1의 page1 결과
        df2 = dfs[1]  # HitterBasic/Basic1의 page2 결과
        df3 = dfs[2]  # HitterBasic/Basic2의 page1 결과
        df4 = dfs[3]  # HitterBasic/Basic2의 page2 결과
        Basic1_df = pd.concat([df1, df2], axis=0)
        Basic2_df = pd.concat([df3, df4], axis=0)
        final_df = pd.concat([Basic1_df, Basic2_df], axis=1)
    else:
        df1 = dfs[0]  # HitterBasic/Basic1의 page1 결과
        df2 = dfs[1]  # HitterBasic/Basic1의 page2 결과
        final_df = pd.concat([df1, df2], axis=1)

    # 열 이름을 한국어로 변환
    final_df.rename(columns=column_translations, inplace=True)

    # 데이터프레임 전체를 문자열로 변환하여 CSV 파일로 저장
    final_df = final_df.astype(str)
    final_df.to_csv(os.path.join(folder_name, f'kiwoom_hitter_2024.csv'), index=False, encoding='cp949')

# 빈 결과 테이블
result_dfs = {year: [] for year in range(2019, 2019 + len(Hitter_years))}

# 사이트 접속
for tep in teps[2:3] :
    url = f'https://www.koreabaseball.com/Record/Player/{tep}.aspx'
    driver.get(url)
    time.sleep(3)

    # 타자 기록 접속
    for year_index, year in enumerate(Hitter_years):
        # 연도 선택
        Season_selector = f'#cphContents_cphContents_cphContents_ddlSeason_ddlSeason > option:nth-child({year})'
        driver.find_element(By.CSS_SELECTOR, Season_selector).click()
        time.sleep(3)
            
        # 팀 정보 선택 ('WO'로 설정)
        team_option = driver.find_element(By.CSS_SELECTOR, f"option[value='{team[0]}']")
        team_option.click()
        time.sleep(3)

        # 1페이지로 돌아가기
        first_page_button = driver.find_element(By.CSS_SELECTOR, f'#cphContents_cphContents_cphContents_ucPager_btnNo1')
        first_page_button.click()
        time.sleep(3)

        # 페이지 반복하여 데이터 가져오기 (1페이지와 2페이지)
        for page in range(1, 3):
            if page > 1:
                try:
                    next_page_button = driver.find_element(By.CSS_SELECTOR, f'#cphContents_cphContents_cphContents_ucPager_btnNo{page}')
                    next_page_button.click()
                    time.sleep(3)
                except:
                    break  # 페이지 버튼이 없는 경우 루프를 종료

            # 결과 테이블 가져오기
            result_table = driver.find_element(By.CSS_SELECTOR, '#cphContents_cphContents_cphContents_udpContent > div.record_result')
            table_html = result_table.getAttribute('outerHTML')

            # DataFrame으로 변환하여 리스트에 추가
            df = pd.read_html(table_html, encoding='utf-8')[0]

            # 결과를 리스트에 추가
            result_dfs[2019 + year_index].append(df)

        # 연도가 바뀔 때마다 다시 첫 페이지로 돌아가기
        driver.get(url)
        time.sleep(3)
        driver.find_element(By.CSS_SELECTOR, Season_selector).click()
        time.sleep(3)

# 각 연도별로 DataFrame을 CSV 파일로 저장
for year, dfs in result_dfs.items():
    # dfs 개수에 따라 다르게 저장
    if len(dfs) > 1 :  
        df1 = dfs[0]  # HitterBasic/Detail1의 page1 결과
        df2 = dfs[1]  # HitterBasic/Detail1c의 page2 결과  
        final_df = pd.concat([df1, df2], axis=0)

    else :
        df1 = dfs[0]  # Runner/Basic의 page1 결과
        final_df = df1

    # 열 이름을 한국어로 변환
    final_df.rename(columns=column_translations, inplace=True)
    
    # 데이터프레임 전체를 문자열로 변환하여 CSV 파일로 저장
    final_df = final_df.astype(str)
    final_df.to_csv(os.path.join(folder_name, f'kiwoom_hitter_detail_2024.csv'), index=False, encoding='cp949')

# 빈 결과 테이블
result_dfs = {year: [] for year in range(2019, 2019 + len(Hitter_years))}

# 사이트 접속
for tep in teps[:1] :
    url = f'https://www.koreabaseball.com/Record/Player/{tep}.aspx'
    driver.get(url)
    time.sleep(3)

    # 타자 기록 접속
    for year_index, year in enumerate(Hitter_years):
        # 연도 선택
        Season_selector = f'#cphContents_cphContents_cphContents_ddlSeason_ddlSeason > option:nth-child({year})'
        driver.find_element(By.CSS_SELECTOR, Season_selector).click()
        time.sleep(3)
            
        # 팀 정보 선택 ('WO'로 설정)
        team_option = driver.find_element(By.CSS_SELECTOR, f"option[value='{team[0]}']")
        team_option.click()
        time.sleep(3)

        # 타순별 선택
        Situation_selector = '#cphContents_cphContents_cphContents_ddlSituation_ddlSituation > option:nth-child(14)'
        driver.find_element(By.CSS_SELECTOR, Situation_selector).click()
        time.sleep(3)

        for order in batting_order:
            # 타순 선택
            Batting_order_selector = f'#cphContents_cphContents_cphContents_ddlSituationDetail_ddlSituationDetail > option:nth-child({order})'
            driver.find_element(By.CSS_SELECTOR, Batting_order_selector).click()
            time.sleep(3)

            # 1페이지로 돌아가기
            first_page_button = driver.find_element(By.CSS_SELECTOR, f'#cphContents_cphContents_cphContents_ucPager_btnNo1')
            first_page_button.click()
            time.sleep(3)

            # 페이지 반복하여 데이터 가져오기 (1페이지와 2페이지)
            for page in range(1, 3):
                if page > 1:
                    try:
                        next_page_button = driver.find_element(By.CSS_SELECTOR, f'#cphContents_cphContents_cphContents_ucPager_btnNo{page}')
                        next_page_button.click()
                        time.sleep(3)
                    except:
                        break  # 페이지 버튼이 없는 경우 루프를 종료

                # 결과 테이블 가져오기
                result_table = driver.find_element(By.CSS_SELECTOR, '#cphContents_cphContents_cphContents_udpContent > div.record_result')
                table_html = result_table.getAttribute('outerHTML')

                # DataFrame으로 변환하여 리스트에 추가
                df = pd.read_html(table_html, encoding='utf-8')[0]

                # 타순 열 추가
                df['타순'] = order_mapping[order]

                # 결과를 리스트에 추가
                result_dfs[2019 + year_index].append(df)

        # 연도가 바뀔 때마다 다시 첫 페이지로 돌아가기
        driver.get(url)
        time.sleep(3)
        driver.find_element(By.CSS_SELECTOR, Season_selector).click()
        time.sleep(3)

# 결과 데이터프레임을 연도별로 합쳐서 각각의 CSV 파일로 저장
for year, dfs in result_dfs.items():
    if dfs:  # dfs 리스트가 비어있지 않을 때만 처리
        combined_df = pd.concat(dfs, axis=0)
        combined_df.rename(columns=column_translations, inplace=True)
        combined_df = combined_df.astype(str)
        combined_df.to_csv(os.path.join(folder_name, f'kiwoom_batting_order_2024.csv'), index=False, encoding='cp949')

# 빈 결과 테이블
result_dfs = {year: [] for year in range(2019, 2019 + len(Hitter_years))}

# 사이트 접속
for tep in teps[3:4]:
    url = f'https://www.koreabaseball.com/Record/Player/{tep}.aspx'
    driver.get(url)
    time.sleep(3)

    # 수비 기록 접속
    for year_index, year in enumerate(Defense_years):
        # 연도 선택
        Season_selector = f'#cphContents_cphContents_cphContents_ddlSeason_ddlSeason > option:nth-child({year})'
        driver.find_element(By.CSS_SELECTOR, Season_selector).click()
        time.sleep(3)
    
        # 팀 정보 선택 ('WO'로 설정)
        team_option = driver.find_element(By.CSS_SELECTOR, f"option[value='{team[0]}']")
        team_option.click()
        time.sleep(3)

        # 1페이지로 돌아가기
        first_page_button = driver.find_element(By.CSS_SELECTOR, f'#cphContents_cphContents_cphContents_ucPager_btnNo1')
        first_page_button.click()
        time.sleep(3)

        # 페이지 반복하여 데이터 가져오기 (1페이지~ 4페이지)
        for page in range(1, 5):
            if page > 1:
                try:
                    next_page_button = driver.find_element(By.CSS_SELECTOR, f'#cphContents_cphContents_cphContents_ucPager_btnNo{page}')
                    next_page_button.click()
                    time.sleep(3)
                except:
                    break  # 페이지 버튼이 없는 경우 루프를 종료

            # 결과 테이블 가져오기
            result_table = driver.find_element(By.CSS_SELECTOR, '#cphContents_cphContents_cphContents_udpContent > div.record_result')
            table_html = result_table.getAttribute('outerHTML')

            # DataFrame으로 변환하여 리스트에 추가
            df = pd.read_html(table_html, encoding='utf-8')[0]

            # 결과를 리스트에 추가
            result_dfs[2019 + year_index].append(df)

        # 팀이 바뀔 때마다 다시 첫 페이지로 돌아가기
        driver.get(url)
        time.sleep(3)
        driver.find_element(By.CSS_SELECTOR, Season_selector).click()
        time.sleep(3)

# 각 연도별로 DataFrame을 CSV 파일로 저장
for year, dfs in result_dfs.items():
    # dfs 개수에 따라 다르게 저장
    if not dfs:
        continue  # dfs 리스트가 비어 있는 경우 반복문을 건너뜁니다.
        
    if len(dfs) >= 4 :
        df1 = dfs[0]  # Defense/Basic의 page1 결과
        df2 = dfs[1]  # Defense/Basic의 page2 결과
        df3 = dfs[2]  # Defense/Basic의 page3 결과
        df4 = dfs[3]  # Defense/Basic의 page4 결과        
        final_df = pd.concat([df1, df2, df3, df4], axis=0)
    elif len(dfs) >= 3 :
        df1 = dfs[0]  # Defense/Basic의 page1 결과
        df2 = dfs[1]  # Defense/Basic의 page2 결과
        df3 = dfs[2]  # Defense/Basic의 page3 결과
        final_df = pd.concat([df1, df2, df3], axis=0)
    else :
        df1 = dfs[0]  # Defense/Basic의 page1 결과
        df2 = dfs[1]  # Defense/Basic의 page2 결과
        final_df = pd.concat([df1, df2], axis=0)

    # 열 이름을 한국어로 변환
    final_df.rename(columns=column_translations, inplace=True)
    
    # 데이터프레임 전체를 문자열로 변환하여 CSV 파일로 저장
    final_df = final_df.astype(str)
    final_df.to_csv(os.path.join(folder_name, f'kiwoom_defense_2024.csv'), index=False, encoding='cp949')

# 빈 결과 테이블
result_dfs = {year: [] for year in range(2019, 2019 + len(Hitter_years))}

# 사이트 접속
for tep in teps[4:] :
    url = f'https://www.koreabaseball.com/Record/Player/{tep}.aspx'
    driver.get(url)
    time.sleep(3)

    # 주루 기록 접속
    for year_index, year in enumerate(Runner_years):
        # 연도 선택
        Season_selector = f'#cphContents_cphContents_cphContents_ddlSeason_ddlSeason > option:nth-child({year})'
        driver.find_element(By.CSS_SELECTOR, Season_selector).click()
        time.sleep(3)

        # 팀 정보 선택 ('WO'로 설정)
        team_option = driver.find_element(By.CSS_SELECTOR, f"option[value='{team[0]}']")
        team_option.click()
        time.sleep(3)

        # 1페이지로 돌아가기
        first_page_button = driver.find_element(By.CSS_SELECTOR, f'#cphContents_cphContents_cphContents_ucPager_btnNo1')
        first_page_button.click()
        time.sleep(3)

        # 페이지 반복하여 데이터 가져오기 (1페이지~ 4페이지)
        for page in range(1, 3):
            if page > 1:
                try:
                    next_page_button = driver.find_element(By.CSS_SELECTOR, f'#cphContents_cphContents_cphContents_ucPager_btnNo{page}')
                    next_page_button.click()
                    time.sleep(3)
                except:
                    break  # 페이지 버튼이 없는 경우 루프를 종료

            # 결과 테이블 가져오기
            result_table = driver.find_element(By.CSS_SELECTOR, '#cphContents_cphContents_cphContents_udpContent > div.record_result')
            table_html = result_table.getAttribute('outerHTML')

            # DataFrame으로 변환하여 리스트에 추가
            df = pd.read_html(table_html, encoding='utf-8')[0]

            # 결과를 리스트에 추가
            result_dfs[2019 + year_index].append(df)

        # 연도가 바뀔 때마다 다시 첫 페이지로 돌아가기
        driver.get(url)
        time.sleep(3)
        driver.find_element(By.CSS_SELECTOR, Season_selector).click()
        time.sleep(3)

# 각 연도별로 DataFrame을 CSV 파일로 저장
for year, dfs in result_dfs.items():
    # dfs 개수에 따라 다르게 저장
    if len(dfs) > 1 :  
        df1 = dfs[0]  # Runner/Basic의 page1 결과
        df2 = dfs[1]  # Runner/Basic의 page2 결과  
        final_df = pd.concat([df1, df2], axis=0)
    else :
        df1 = dfs[0]  # Runner/Basic의 page1 결과
        final_df = df1

    # 열 이름을 한국어로 변환
    final_df.rename(columns=column_translations, inplace=True)
    
    # 데이터프레임 전체를 문자열로 변환하여 CSV 파일로 저장
    final_df = final_df.astype(str)
    final_df.to_csv(os.path.join(folder_name, f'kiwoom_runner_2024.csv'), index=False, encoding='cp949')
        
# 드라이버 종료
driver.quit()

# Git 명령어를 사용하여 변경 사항을 추가, 커밋, 푸시합니다.
def git_commit_and_push(commit_message):
    try:
        # Git 상태 확인
        subprocess.run(["git", "status"], check=True)
        # 변경된 파일 스테이징
        subprocess.run(["git", "add", "."], check=True)
        # 커밋
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        # 원격 저장소에 푸시
        subprocess.run(["git", "push", "origin", "main"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing git command: {e}")

# 커밋 메시지 설정
commit_message = f"Add CSV files for {current_date}"

# Git 커밋 및 푸시 함수 호출
git_commit_and_push(commit_message)
